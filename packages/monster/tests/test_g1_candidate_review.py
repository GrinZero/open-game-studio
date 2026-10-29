"""Candidate-video-first review must survive old callers, incomplete evidence and gate use."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/monster-workflow/scripts'))
import sop_g1_pack as g1
import sop_gate as gate


class VideoLayout(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.depth = 0
        self.videos = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            self.depth += 1
        if tag == 'video':
            self.videos.append((dict(attrs), self.depth))

    def handle_endtag(self, tag):
        if tag == 'details':
            self.depth -= 1


class CandidateReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(cls.temp.name)
        ffmpeg = g1._find_ffmpeg_binary()
        if not ffmpeg:
            raise unittest.SkipTest('requires ffmpeg')
        cls.colors = [(180, 30, 30), (30, 180, 30), (30, 30, 180), (180, 180, 30)]
        for index in range(8):
            frame = Image.new('RGB', (1280, 720))
            draw = ImageDraw.Draw(frame)
            for cid, color in enumerate(cls.colors):
                x, y = (cid % 2) * 640, (cid // 2) * 360
                draw.rectangle((x, y, x + 639, y + 359), fill=tuple(c + index * 3 for c in color))
            frame.save(cls.folder / f'frame_{index:04d}.png')
        # Fractional FPS ensures crop encoding does not round or resample the timeline.
        source = cls.folder / 'sync.mp4'
        subprocess.run([ffmpeg, '-y', '-framerate', '24000/1001', '-i',
                        str(cls.folder / 'frame_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        str(source)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cls.sources = {view: source for view in g1.G1_SYNC_VIEWS}
        cls.videos = g1.encode_g1_candidate_videos(cls.sources, cls.folder)
        cls.issues = [dict(candidate_id=i, status='FAIL', check='stretch', detail='real failure',
                           metrics={'max_edge_stretch_ratio': 10}) for i in range(4)]
        cls.page = cls.folder / 'index.html'
        cls.build_page(cls.page, candidate_video_rels=cls.videos)

    @classmethod
    def build_page(cls, path, **kwargs):
        return g1.build_g1_review_html('fixture', 'quadruped', [], {},
                                      {v: str(p) for v, p in cls.sources.items()}, cls.issues, path, **kwargs)

    def report(self, contract=True):
        def evidence(path):
            return {'path': str(path), 'sha256': g1.sha256_file(path)}
        blend = self.folder / 'fixture.blend'
        blend.write_bytes(b'BLENDER-v402' + b'\0' * 2048)
        meta = self.folder / 'meta.json'
        meta.write_text('{}')
        png = self.folder / 'bones.png'
        Image.new('RGB', (640, 640)).save(png)
        issues = self.folder / 'issues.json'
        issues.write_text(json.dumps(self.issues))
        result = dict(schema_version=2, backend='blender', passed=True, review_html=str(self.page), evidence={
            'candidate_inputs': {'0': evidence(meta)}, 'candidate_blends': {'0': evidence(blend)},
            'candidate_metas': {'0': evidence(meta)}, 'octahedral_views': {'0': evidence(png)},
            'sync_videos': {v: evidence(p) for v, p in self.sources.items()},
            'follow_through_issues': evidence(issues), 'review_html': evidence(self.page)})
        if contract:
            result.update(presentation_contract=g1.G1_REVIEW_CONTRACT, candidate_videos=copy.deepcopy(self.videos),
                          video_timing=dict(fps=24000/1001, frames_by_view={v: 8 for v in g1.G1_SYNC_VIEWS},
                                            transform='quadrant_crop_only_no_retiming'))
            result['evidence']['candidate_videos'] = {
                f'candidate_{cid}_{view}': {**evidence(self.folder / rel),
                    'source_sync_sha256': g1.sha256_file(self.sources[view]),
                    'crop_xywh': [(cid % 2) * 640, (cid // 2) * 360, 640, 360]}
                for cid, views in self.videos.items() for view, rel in views.items()}
        return result

    def test_real_crop_preserves_all_quadrants_frames_and_fractional_fps(self):
        probe = shutil.which('ffprobe')
        if not probe:
            self.skipTest('requires ffprobe')
        for cid, views in self.videos.items():
            for rel in views.values():
                path = self.folder / rel
                stream = json.loads(subprocess.check_output([probe, '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(path)]))['streams'][0]
                self.assertEqual((stream['width'], stream['height']), (640, 360))
                self.assertEqual(stream['r_frame_rate'], '24000/1001')
                self.assertEqual(int(stream['nb_frames']), 8)
                self.assertAlmostEqual(float(stream['duration']), 8 * 1001 / 24000, places=5)
                pixels = subprocess.check_output([g1._find_ffmpeg_binary(), '-v', 'error', '-i', str(path),
                    '-vf', 'scale=1:1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
                self.assertEqual(len(pixels), 8 * 3)
                for frame in (0, 7):
                    for actual, expected in zip(pixels[frame * 3:frame * 3 + 3], self.colors[cid]):
                        self.assertLess(abs(actual - (expected + frame * 3)), 8)

    def test_page_shows_four_independent_cards_before_folded_comparison_and_all_fails(self):
        text = self.page.read_text()
        layout = VideoLayout(text)
        self.assertEqual([v['id'] for v, depth in layout.videos if depth == 0],
                         [f'candidate-video-{i}' for i in range(4)])
        self.assertEqual(sum(depth > 0 for _, depth in layout.videos), 3)
        self.assertEqual(text.count('<option '), 12)
        self.assertEqual(text.count('自动检查：FAIL'), 4)
        self.assertIn('不能进入 G2', text)
        for attrs, depth in layout.videos:
            self.assertIn('controls', attrs)

    def test_old_html_caller_generates_individuals_and_missing_source_fails(self):
        with tempfile.TemporaryDirectory(dir=self.folder) as tmp:
            page = self.build_page(Path(tmp) / 'index.html')
            self.assertEqual(len(list(Path(tmp).glob('candidate_*.mp4'))), 12)
            self.assertEqual(sum(depth == 0 for _, depth in VideoLayout(page.read_text()).videos), 4)
        with self.assertRaises(RuntimeError):
            g1.encode_g1_candidate_videos({'front': self.sources['front']}, self.folder / 'incomplete')

    def test_partial_duplicate_and_missing_video_cannot_fall_back(self):
        variants = [{}, {0: self.videos[0]}, {**self.videos, 0: {'side': 'one.mp4'}},
                    {i: {v: 'same.mp4' for v in g1.G1_SYNC_VIEWS} for i in range(4)},
                    {**self.videos, 0: {**self.videos[0], 'front': 'missing.mp4'}}]
        for videos in variants:
            with self.subTest(videos=videos), self.assertRaises((RuntimeError, FileNotFoundError)):
                self.build_page(self.folder / 'invalid.html', candidate_video_rels=videos)

    def test_report_requires_complete_bound_evidence_and_preserves_legacy(self):
        report = self.report()
        g1.validate_g1_evidence(report)
        g1.validate_g1_evidence(self.report(contract=False))
        for field in ('candidate_videos', 'presentation_contract'):
            bad = copy.deepcopy(report)
            if field == 'candidate_videos':
                del bad[field]
            else:
                bad[field] = 'unknown-contract'
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                g1.validate_g1_evidence(bad)
        for mutation in ('missing', 'hash', 'reference', 'crop', 'source', 'frames'):
            bad = copy.deepcopy(report)
            item = bad['evidence']['candidate_videos']['candidate_0_front']
            if mutation == 'missing': del bad['evidence']['candidate_videos']['candidate_3_side']
            elif mutation == 'hash': item['sha256'] = 'invalid'
            elif mutation == 'reference': bad['candidate_videos'][0]['front'] = 'different.mp4'
            elif mutation == 'crop': item['crop_xywh'][0] = 640
            elif mutation == 'source': item['source_sync_sha256'] = 'invalid'
            else: bad['video_timing']['frames_by_view']['side'] = 7
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                g1.validate_g1_evidence(bad)

    def test_submission_approval_and_downstream_recheck_bound_report(self):
        report_path = self.folder / 'report.json'
        report_path.write_text(json.dumps(self.report()))
        ledger = {'gates': {}, 'fingerprints': {}}
        materials = dict(presentation_contract=g1.G1_REVIEW_CONTRACT, g1_report_json=str(report_path))
        info = gate.mark_awaiting_review(ledger, 'G1', review_pack_path=str(self.page), materials=materials)
        self.assertEqual(info['user_decision'], 'pending')
        report_path.write_text(report_path.read_text() + '\n')
        # Simulated approval state exercises downstream rejection without recording any real decision.
        info.update(status='approved', user_decision='approved', bone_naming_validation={'valid': True})
        with self.assertRaisesRegex(RuntimeError, '报告已改变'):
            gate.assert_gate(ledger, 'G1')
        with self.assertRaisesRegex(RuntimeError, '报告已改变'):
            gate.record_user_decision(ledger, 'G1', decision='approved', user_quote='test', selected_candidate_id=0)


if __name__ == '__main__':
    unittest.main()
