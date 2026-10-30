extends GLTFDocumentExtension
## Runtime GLB loading adapter. Probe before use; preserve source KTX2 bytes.

# Synthetic 4x4 ETC1S sRGB gray, independently decoded to RGB (128,128,128).
const PROBE_BASE64 := "q0tUWCAyMLsNChoKAAAAAAEAAAAEAAAABAAAAAAAAAAAAAAAAQAAAAEAAAABAAAAaAAAACwAAACUAAAAeAAAABABAAAAAAAAgAAAAAAAAACQAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAsAAAAAAAAAAIAKACjAQIAAwMAAAgAAAAAAAAAAAA/AAAAAAAAAAAA/////xIAAABLVFhvcmllbnRhdGlvbgByZAAAACcAAABLVFh3cml0ZXIAdG9rdHggdjQuNC4yIC8gbGlia3R4IHY0LjQuMgAALgAAAEtUWHdyaXRlclNjUGFyYW1zAC0tZW5jb2RlIGV0YzFzIC0tcWxldmVsIDEyOAAAAAAAAAABAAEAKAAAAAUAAAArAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAcAEAAAAAAAAAgSYCAAAAAAAQFIEABMAAAAAAAAIAWACQAAAAAAAMVRVVVUFAMFEAAAAAAAA8l8tAJgAAAAAAABACAATAAIAAAAAiAHABAAAAAAAAAIIAAA="
const PROBE_SRGB := Vector3(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0)

# Exported configuration survives GLTFDocument's duplication of extensions.
@export var compensate_prelinearized_srgb: bool = false

func configure_for_current_loader() -> Dictionary:
	compensate_prelinearized_srgb = false
	var image := Image.new()
	var error := image.load_ktx_from_buffer(Marshalls.base64_to_raw(PROBE_BASE64))
	if error != OK:
		return {"ok": false, "error": error, "mode": "ktx_loader_unavailable"}
	if image.is_compressed():
		error = image.decompress()
		if error != OK:
			return {"ok": false, "error": error, "mode": "probe_decompression_unavailable"}
	var pixel := image.get_pixel(0, 0)
	var observed := Vector3(pixel.r, pixel.g, pixel.b)
	var expected_linear_color := Color(PROBE_SRGB.x, PROBE_SRGB.y, PROBE_SRGB.z).srgb_to_linear()
	var expected_linear := Vector3(expected_linear_color.r, expected_linear_color.g, expected_linear_color.b)
	var mode := ""
	if observed.distance_to(PROBE_SRGB) <= 0.015:
		mode = "passthrough"
	elif observed.distance_to(expected_linear) <= 0.015:
		mode = "compensate_prelinearized_srgb"
		compensate_prelinearized_srgb = true
	else:
		return {"ok": false, "error": ERR_INVALID_DATA, "mode": "unrecognized_color_transfer", "observed_rgb": [pixel.r, pixel.g, pixel.b]}
	return {"ok": true, "error": OK, "mode": mode, "observed_rgb": [pixel.r, pixel.g, pixel.b], "engine": Engine.get_version_info().string, "renderer": RenderingServer.get_current_rendering_method()}

func load_glb(path: String) -> Dictionary:
	var probe := configure_for_current_loader()
	if not probe.ok:
		return {"error": probe.error, "scene": null, "loader": probe}
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	# Registration is global. Keep this synchronous and do not yield in this block.
	if compensate_prelinearized_srgb:
		GLTFDocument.register_gltf_document_extension(self, true)
	var error := document.append_from_file(path, state)
	if compensate_prelinearized_srgb:
		GLTFDocument.unregister_gltf_document_extension(self)
	if error != OK:
		return {"error": error, "scene": null, "loader": probe}
	var scene := document.generate_scene(state)
	if scene == null:
		return {"error": ERR_CANT_CREATE, "scene": null, "loader": probe}
	return {"error": OK, "scene": scene, "loader": probe, "acceptance": "awaiting_user_comparison"}

func _import_preflight(_state: GLTFState, extensions: PackedStringArray) -> Error:
	return OK if compensate_prelinearized_srgb and extensions.has("KHR_texture_basisu") else ERR_SKIP

func _get_supported_extensions() -> PackedStringArray:
	return PackedStringArray(["KHR_texture_basisu"])

func _parse_image_data(_state: GLTFState, bytes: PackedByteArray, mime: String, image: Image) -> Error:
	if mime != "image/ktx2" or not compensate_prelinearized_srgb:
		return OK # Leave the image empty so other registered loaders can handle it.
	if bytes.size() < 80 or bytes.slice(0, 12) != PackedByteArray([171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10]):
		return ERR_FILE_CORRUPT
	var dfd := bytes.decode_u32(48)
	if dfd + 16 > bytes.size():
		return ERR_FILE_CORRUPT
	if bytes[dfd + 14] != 2: # Only DFD sRGB color images; linear data maps pass through.
		return OK
	var error := image.load_ktx_from_buffer(bytes)
	if error != OK:
		return error
	if image.get_format() not in [Image.FORMAT_RGB8, Image.FORMAT_RGBA8]:
		return ERR_UNAVAILABLE
	# Undo the detected loader conversion before StandardMaterial's color sampling.
	image.linear_to_srgb()
	return OK

func _get_image_file_extension() -> String:
	return "" # Editor integration is not automatic; runtime load_glb is tested here.
