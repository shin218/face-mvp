package com.example.demo.controller;

import com.example.demo.service.FacePlusService;
import com.example.demo.service.FalInstantIdService;
import com.example.demo.service.MultiModelVibeService;
import com.example.demo.service.vibe.ModelAfterResult;
import com.example.demo.service.vibe.ModelVibeResult;
import com.example.demo.service.vibe.VibeInput;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;
import java.util.Base64;

@Controller
public class FaceController {

	private static final long MAX_FILE_BYTES = 10L * 1024L * 1024L;
	private static final Set<String> ALLOWED_EXT = Set.of("jpg", "jpeg", "png");
	private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of(
			MediaType.IMAGE_JPEG_VALUE,
			MediaType.IMAGE_PNG_VALUE
	);

	private final FacePlusService facePlusService;
	private final MultiModelVibeService multiModelVibeService;
	private final FalInstantIdService falInstantIdService;
	private final boolean falEnabled;

	public FaceController(
			FacePlusService facePlusService,
			MultiModelVibeService multiModelVibeService,
			FalInstantIdService falInstantIdService,
			@Value("${features.fal-enabled:false}") boolean falEnabled
	) {
		this.facePlusService = facePlusService;
		this.multiModelVibeService = multiModelVibeService;
		this.falInstantIdService = falInstantIdService;
		this.falEnabled = falEnabled;
	}

	@GetMapping("/")
	public String index(Model model) {
		if (!model.containsAttribute("assessment")) {
			model.addAttribute("assessment", null);
		}
		if (!model.containsAttribute("vibeResults")) {
			model.addAttribute("vibeResults", Collections.emptyList());
		}
		model.addAttribute("beforeImageDataUri", null);
		model.addAttribute("ref1DataUri", null);
		model.addAttribute("ref2DataUri", null);
		model.addAttribute("afterResultMap", Collections.emptyMap());
		return "upload";
	}

	@PostMapping(path = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public String upload(
			@RequestParam("photo") MultipartFile photo,
			@RequestParam(value = "ref1", required = false) MultipartFile ref1,
			@RequestParam(value = "ref2", required = false) MultipartFile ref2,
			@RequestParam(value = "vibeText", required = false) String vibeText,
			Model model
	) {
		try {
			// 고객 사진 검증 (필수)
			if (photo == null || photo.isEmpty()) {
				throw new IllegalArgumentException("고객 사진을 선택해주세요.");
			}
			validateImageFile(photo, "고객 사진");
			byte[] photoBytes = photo.getBytes();
			String photoMime = toMime(photo.getContentType());

			// 레퍼런스 1 (선택)
			byte[] ref1Bytes = null;
			String ref1Mime = null;
			if (ref1 != null && !ref1.isEmpty()) {
				validateImageFile(ref1, "레퍼런스 1");
				ref1Bytes = ref1.getBytes();
				ref1Mime = toMime(ref1.getContentType());
			}

			// 레퍼런스 2 (선택)
			byte[] ref2Bytes = null;
			String ref2Mime = null;
			if (ref2 != null && !ref2.isEmpty()) {
				validateImageFile(ref2, "레퍼런스 2");
				ref2Bytes = ref2.getBytes();
				ref2Mime = toMime(ref2.getContentType());
			}

			// 썸네일 data URI
			model.addAttribute("beforeImageDataUri",
					"data:" + photoMime + ";base64," + Base64.getEncoder().encodeToString(photoBytes));
			model.addAttribute("ref1DataUri",
					ref1Bytes != null ? "data:" + ref1Mime + ";base64," + Base64.getEncoder().encodeToString(ref1Bytes) : null);
			model.addAttribute("ref2DataUri",
					ref2Bytes != null ? "data:" + ref2Mime + ";base64," + Base64.getEncoder().encodeToString(ref2Bytes) : null);
			model.addAttribute("vibeTextEcho", vibeText);

			// Face++ 분석
			var assessment = facePlusService.assess(photoBytes, photo.getOriginalFilename());
			model.addAttribute("assessment", assessment);

			// VibeInput 구성
			var input = new VibeInput();
			input.customerImageBytes = photoBytes;
			input.customerImageMime = photoMime;
			input.ref1Bytes = ref1Bytes;
			input.ref1Mime = ref1Mime;
			input.ref2Bytes = ref2Bytes;
			input.ref2Mime = ref2Mime;
			input.vibeText = vibeText;

			input.detectPrettyJson = assessment.prettyJson;
			input.gender = assessment.gender == null ? null : assessment.gender.value;
			input.age = assessment.age == null ? null : (int) Math.round(assessment.age.value);
			input.emotionSummary = assessment.emotionSummary;
			input.skinHealth = assessment.skinstatus == null ? 0.0 : assessment.skinstatus.health;
			input.skinStain = assessment.skinstatus == null ? 0.0 : assessment.skinstatus.stain;
			input.skinAcne = assessment.skinstatus == null ? 0.0 : assessment.skinstatus.acne;
			input.skinDarkCircle = assessment.skinstatus == null ? 0.0 : assessment.skinstatus.darkCircle;
			input.smile = assessment.smile == null ? null : assessment.smile.value;
			input.glass = assessment.glass == null ? null : assessment.glass.value;
			input.eyeDistancePx = (assessment.structure == null) ? null : assessment.structure.eyeDistance;
			input.faceWidthHeightRatio = (assessment.structure == null) ? null : assessment.structure.faceWidthHeightRatio;
			input.noseChinToForeheadNoseRatio = (assessment.structure == null) ? null : assessment.structure.noseChinToForeheadNoseRatio;
			input.leftEyeWidthPx = (assessment.structure == null) ? null : assessment.structure.leftEyeWidthPx;
			input.rightEyeWidthPx = (assessment.structure == null) ? null : assessment.structure.rightEyeWidthPx;
			input.lipThicknessPx = (assessment.structure == null) ? null : assessment.structure.lipThicknessPx;
			input.faceShape = (assessment.structure == null) ? null : assessment.structure.faceShape;
			input.bangsLikely = (assessment.structure == null) ? null : assessment.structure.bangsLikely;

			List<ModelVibeResult> vibeResults = multiModelVibeService.analyzeParallel(input);
			model.addAttribute("vibeResults", vibeResults);

			// After 이미지 (fal 비활성화 시 스킵)
			Map<String, ModelAfterResult> afterResultMap = falEnabled
					? generateAfterImages(vibeResults, photoBytes, photo.getContentType())
					: Collections.emptyMap();
			model.addAttribute("afterResultMap", afterResultMap);

			return "upload";
		} catch (Exception e) {
			model.addAttribute("error", e.getMessage() == null ? "처리 중 오류가 발생했습니다." : e.getMessage());
			model.addAttribute("assessment", null);
			model.addAttribute("vibeResults", Collections.emptyList());
			model.addAttribute("beforeImageDataUri", null);
			model.addAttribute("ref1DataUri", null);
			model.addAttribute("ref2DataUri", null);
			model.addAttribute("afterResultMap", Collections.emptyMap());
			return "upload";
		}
	}

	private void validateImageFile(MultipartFile file, String label) {
		if (file.getSize() > MAX_FILE_BYTES) {
			throw new IllegalArgumentException(label + " 파일 크기는 10MB 이하만 허용됩니다.");
		}
		String ext = getExtension(file.getOriginalFilename());
		if (ext == null || !ALLOWED_EXT.contains(ext)) {
			throw new IllegalArgumentException(label + ": 허용 확장자는 jpg/jpeg/png입니다.");
		}
		String ct = file.getContentType();
		if (ct == null || !ALLOWED_CONTENT_TYPES.contains(ct)) {
			throw new IllegalArgumentException(label + ": 허용 확장자는 jpg/jpeg/png입니다.");
		}
	}

	private static String toMime(String contentType) {
		if (contentType == null) return "image/jpeg";
		return contentType.contains("png") ? "image/png" : "image/jpeg";
	}

	private Map<String, ModelAfterResult> generateAfterImages(
			List<ModelVibeResult> vibeResults, byte[] bytes, String contentType) {

		if (!falInstantIdService.isConfigured()) {
			return vibeResults.stream().collect(Collectors.toMap(
					ModelVibeResult::getProvider,
					r -> new ModelAfterResult(r.getProvider(), null, null, "FAL_API_KEY 미설정"),
					(a, b) -> a,
					LinkedHashMap::new
			));
		}

		List<CompletableFuture<ModelAfterResult>> futures = vibeResults.stream()
				.map(r -> CompletableFuture.supplyAsync(() ->
						falInstantIdService.generateForModel(
								r.getProvider(), bytes, contentType, r.getText(), r.hasError())))
				.collect(Collectors.toList());

		CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

		Map<String, ModelAfterResult> result = new LinkedHashMap<>();
		for (CompletableFuture<ModelAfterResult> f : futures) {
			ModelAfterResult afterResult = f.join();
			result.put(afterResult.getProvider(), afterResult);
		}
		return result;
	}

	private static String getExtension(String filename) {
		if (filename == null) return null;
		int idx = filename.lastIndexOf('.');
		if (idx < 0 || idx == filename.length() - 1) return null;
		return filename.substring(idx + 1).toLowerCase(Locale.ROOT);
	}
}
