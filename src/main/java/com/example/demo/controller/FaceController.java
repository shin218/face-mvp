package com.example.demo.controller;

import com.example.demo.service.FacePlusService;
import com.example.demo.service.FalInstantIdService;
import com.example.demo.service.MultiModelVibeService;
import com.example.demo.service.vibe.ModelAfterResult;
import com.example.demo.service.vibe.ModelVibeResult;
import com.example.demo.service.vibe.VibeInput;
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

	private static final long MAX_BYTES = 10L * 1024L * 1024L;
	private static final Set<String> ALLOWED_EXT = Set.of("jpg", "jpeg", "png");
	private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of(
			MediaType.IMAGE_JPEG_VALUE,
			MediaType.IMAGE_PNG_VALUE
	);

	private final FacePlusService facePlusService;
	private final MultiModelVibeService multiModelVibeService;
	private final FalInstantIdService falInstantIdService;

	public FaceController(FacePlusService facePlusService,
	                      MultiModelVibeService multiModelVibeService,
	                      FalInstantIdService falInstantIdService) {
		this.facePlusService = facePlusService;
		this.multiModelVibeService = multiModelVibeService;
		this.falInstantIdService = falInstantIdService;
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
		model.addAttribute("afterResultMap", Collections.emptyMap());
		return "upload";
	}

	@PostMapping(path = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public String upload(@RequestParam("photo") MultipartFile photo, Model model) {
		try {
			if (photo == null || photo.isEmpty()) {
				throw new IllegalArgumentException("파일을 선택해주세요.");
			}
			if (photo.getSize() > MAX_BYTES) {
				throw new IllegalArgumentException("파일 크기는 10MB 이하만 허용됩니다.");
			}

			String originalFilename = photo.getOriginalFilename();
			String ext = getExtension(originalFilename);
			if (ext == null || !ALLOWED_EXT.contains(ext)) {
				throw new IllegalArgumentException("업로드 가능한 파일은 jpg, jpeg, png 만 허용됩니다.");
			}

			String contentType = photo.getContentType();
			if (contentType == null || !ALLOWED_CONTENT_TYPES.contains(contentType)) {
				throw new IllegalArgumentException("업로드 가능한 파일은 jpg, jpeg, png 만 허용됩니다.");
			}

			byte[] bytes = photo.getBytes();

			// before 이미지 data URI
			String mimeType = contentType.contains("png") ? "image/png" : "image/jpeg";
			model.addAttribute("beforeImageDataUri",
					"data:" + mimeType + ";base64," + Base64.getEncoder().encodeToString(bytes));

			var assessment = facePlusService.assess(bytes, originalFilename);
			model.addAttribute("assessment", assessment);

			var input = new VibeInput();
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

			// 모델별 after 이미지 병렬 생성
			Map<String, ModelAfterResult> afterResultMap = generateAfterImages(vibeResults, bytes, contentType);
			model.addAttribute("afterResultMap", afterResultMap);

			return "upload";
		} catch (Exception e) {
			model.addAttribute("error", e.getMessage() == null ? "처리 중 오류가 발생했습니다." : e.getMessage());
			model.addAttribute("assessment", null);
			model.addAttribute("vibeResults", Collections.emptyList());
			model.addAttribute("beforeImageDataUri", null);
			model.addAttribute("afterResultMap", Collections.emptyMap());
			return "upload";
		}
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

		// 3개 모델 병렬 생성
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
