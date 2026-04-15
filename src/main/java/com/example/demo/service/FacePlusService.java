package com.example.demo.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.MissingNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.util.UriComponentsBuilder;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import reactor.netty.http.client.HttpClient;
import io.netty.channel.ChannelOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeoutException;

@Service
public class FacePlusService {

	private static final Logger log = LoggerFactory.getLogger(FacePlusService.class);

	private static final String DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect";
	private static final String LANDMARK_URL = "https://api-us.faceplusplus.com/facepp/v3/face/landmark";
	private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(60);

	private final WebClient webClient;
	private final ObjectMapper objectMapper;

	private final String apiKey;
	private final String apiSecret;

	public FacePlusService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${face.api.key}") String apiKey,
			@Value("${face.api.secret}") String apiSecret
	) {
		HttpClient httpClient = HttpClient.create()
				.option(ChannelOption.CONNECT_TIMEOUT_MILLIS, (int) CONNECT_TIMEOUT.toMillis())
				.responseTimeout(RESPONSE_TIMEOUT);

		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.codecs(cfg -> cfg.defaultCodecs().maxInMemorySize(4 * 1024 * 1024))
				.build();
		this.objectMapper = objectMapper;
		this.apiKey = apiKey;
		this.apiSecret = apiSecret;
	}

	public FaceAssessment assess(byte[] imageBytes, String originalFilename) {
		try {
			MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
			bodyBuilder.part("api_key", apiKey);
			bodyBuilder.part("api_secret", apiSecret);
			bodyBuilder.part("return_attributes", "headpose,blur,eyestatus,smiling,gender,age,emotion,beauty,skinstatus,facequality");
			bodyBuilder.part("image_file", imageBytes)
					.filename(originalFilename == null ? "upload.jpg" : originalFilename)
					.contentType(MediaType.APPLICATION_OCTET_STREAM);

			MultiValueMap<String, org.springframework.http.HttpEntity<?>> multipart = bodyBuilder.build();

			String url = UriComponentsBuilder.fromUriString(DETECT_URL).build().toUriString();

			String raw = webClient.post()
					.uri(url)
					.contentType(MediaType.MULTIPART_FORM_DATA)
					.body(BodyInserters.fromMultipartData(multipart))
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.onErrorMap(TimeoutException.class, ex ->
							new RuntimeException("Face++ 응답이 지연되어 타임아웃이 발생했습니다. (최대 " + RESPONSE_TIMEOUT.toSeconds() + "초)", ex)
					)
					.onErrorMap(ex -> !(ex instanceof RuntimeException), ex ->
							new RuntimeException("Face++ 호출 중 네트워크 오류가 발생했습니다: " + ex.getClass().getSimpleName(), ex)
					)
					.block();

			if (raw == null || raw.isBlank()) {
				throw new RuntimeException("Face++ 응답이 비어있습니다.");
			}

			JsonNode json = objectMapper.readTree(raw);
			String pretty = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(json);
			DetectResponse dto = toDto(json);
			return evaluate(dto, raw, pretty);
		} catch (WebClientResponseException e) {
			String body = e.getResponseBodyAsString();
			String msg = "Face++ 호출 실패 (" + e.getStatusCode().value() + ")";
			if (body != null && !body.isBlank()) {
				msg += ": " + body;
			}
			throw new RuntimeException(msg, e);
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage() == null ? "Face++ 호출 중 오류가 발생했습니다." : e.getMessage(), e);
		}
	}

	private DetectResponse toDto(JsonNode root) {
		DetectResponse resp = new DetectResponse();
		resp.requestId = text(root, "request_id");
		resp.imageId = text(root, "image_id");

		JsonNode faces = root.path("faces");
		if (faces.isArray()) {
			for (JsonNode f : faces) {
				Face face = new Face();
				face.faceToken = text(f, "face_token");
				face.faceRectangleWidth = num(f.path("face_rectangle"), "width");
				face.faceRectangleHeight = num(f.path("face_rectangle"), "height");

				JsonNode attributes = f.path("attributes");
				if (!attributes.isMissingNode()) {
					FaceAttributes attrs = new FaceAttributes();

					JsonNode headpose = attributes.path("headpose");
					if (!headpose.isMissingNode()) {
						Headpose hp = new Headpose();
						hp.yawAngle = num(headpose, "yaw_angle");
						hp.pitchAngle = num(headpose, "pitch_angle");
						hp.rollAngle = num(headpose, "roll_angle");
						attrs.headpose = hp;
					}

					JsonNode blur = attributes.path("blur");
					if (!blur.isMissingNode()) {
						Blur b = new Blur();
						b.blurnessValue = num(blur.path("blurness"), "value");
						b.blurnessThreshold = num(blur.path("blurness"), "threshold");
						b.motionblurValue = num(blur.path("motionblur"), "value");
						b.motionblurThreshold = num(blur.path("motionblur"), "threshold");
						b.gaussianblurValue = num(blur.path("gaussianblur"), "value");
						b.gaussianblurThreshold = num(blur.path("gaussianblur"), "threshold");
						attrs.blur = b;
					}

					JsonNode eye = attributes.path("eyestatus");
					if (!eye.isMissingNode()) {
						Eyestatus es = new Eyestatus();
						es.leftEyeOpen = bestOpenProb(eye.path("left_eye_status"));
						es.rightEyeOpen = bestOpenProb(eye.path("right_eye_status"));
						attrs.eyestatus = es;
					}

					JsonNode smile = attributes.path("smile");
					if (!smile.isMissingNode()) {
						Smile s = new Smile();
						s.value = num(smile, "value");
						s.threshold = num(smile, "threshold");
						attrs.smile = s;
					}

					JsonNode glass = attributes.path("glass");
					if (!glass.isMissingNode()) {
						Glass g = new Glass();
						g.value = text(glass, "value");
						attrs.glass = g;
					}

					JsonNode gender = attributes.path("gender");
					if (!gender.isMissingNode()) {
						Gender g = new Gender();
						g.value = text(gender, "value");
						g.confidence = num(gender, "confidence");
						attrs.gender = g;
					}

					JsonNode age = attributes.path("age");
					if (!age.isMissingNode()) {
						Age a = new Age();
						a.value = num(age, "value");
						attrs.age = a;
					}

					JsonNode emotion = attributes.path("emotion");
					if (!emotion.isMissingNode()) {
						attrs.emotionJson = emotion;
						attrs.emotionSummary = summarizeEmotion(emotion);
					}

					JsonNode beauty = attributes.path("beauty");
					if (!beauty.isMissingNode()) {
						Beauty b = new Beauty();
						b.maleScore = num(beauty, "male_score");
						b.femaleScore = num(beauty, "female_score");
						attrs.beauty = b;
					}

					JsonNode skin = attributes.path("skinstatus");
					if (!skin.isMissingNode()) {
						Skinstatus s = new Skinstatus();
						s.health = num(skin, "health");
						s.stain = num(skin, "stain");
						s.acne = num(skin, "acne");
						s.darkCircle = num(skin, "dark_circle");
						attrs.skinstatus = s;
					}

					JsonNode fq = attributes.path("facequality");
					if (!fq.isMissingNode()) {
						Facequality q = new Facequality();
						q.value = num(fq, "value");
						q.threshold = num(fq, "threshold");
						attrs.facequality = q;
					}

					face.attributes = attrs;
				}

				resp.faces.add(face);
			}
		}
		return resp;
	}

	private FaceAssessment evaluate(DetectResponse dto, String rawJson, String prettyJson) {
		int facesCount = dto.faces.size();
		if (facesCount == 0) {
			return FaceAssessment.fail("얼굴을 찾지 못했습니다", dto, rawJson, prettyJson, null);
		}
		if (facesCount >= 2) {
			return FaceAssessment.fail("사진에는 1명만 나오게 해주세요", dto, rawJson, prettyJson, null);
		}

		Face first = dto.faces.get(0);
		FaceAttributes attrs = first.attributes;

		if (log.isDebugEnabled() && first.faceToken != null) {
			String ft = first.faceToken;
			String th = ft.length() <= 10 ? ft : (ft.substring(0, 6) + "…" + ft.substring(ft.length() - 4));
			log.debug("Detect → Landmark 연결: face_token hint={}, len={}", th, ft.length());
		}

		FaceStructure structure = null;
		try {
			structure = fetchAndComputeStructure(first, attrs);
			if (structure == null) {
				log.warn("Landmark 기반 structure 가 null 입니다. face_token_present={}, token_len={}",
						first.faceToken != null && !first.faceToken.isBlank(),
						first.faceToken == null ? 0 : first.faceToken.length());
			} else {
				log.debug("Landmark 계산 완료: eyeDistance={}, faceWH={}, leftEyeW={}, rightEyeW={}, midRatio={}, lip={}, shape={}, bangs={}",
						structure.eyeDistance, structure.faceWidthHeightRatio,
						structure.leftEyeWidthPx, structure.rightEyeWidthPx,
						structure.noseChinToForeheadNoseRatio, structure.lipThicknessPx,
						structure.faceShape, structure.bangsLikely);
			}
		} catch (Exception ex) {
			log.warn("Landmark/structure 계산 중 예외 (Detect/판정은 계속): {}", ex.toString(), ex);
		}

		// 하드코딩 기준값 (테스트용)
		double headposeAngleMax = 15.0; // degrees
		double blurValueMax = 10.0;     // 0~100 scale-ish
		double eyeOpenMin = 50.0;       // probability-ish

		if (attrs != null && attrs.headpose != null) {
			Headpose hp = attrs.headpose;
			if (abs(hp.yawAngle) > headposeAngleMax || abs(hp.pitchAngle) > headposeAngleMax || abs(hp.rollAngle) > headposeAngleMax) {
				return FaceAssessment.retry("고개 각도가 너무 큽니다. 정면을 바라보게 다시 촬영해주세요", dto, rawJson, prettyJson, structure);
			}
		}

		if (attrs != null && attrs.blur != null) {
			Blur b = attrs.blur;
			if (b.blurnessValue > blurValueMax || b.motionblurValue > blurValueMax || b.gaussianblurValue > blurValueMax) {
				return FaceAssessment.retry("사진이 흐립니다. 흔들림 없이 다시 촬영해주세요", dto, rawJson, prettyJson, structure);
			}
		}

		if (attrs != null && attrs.eyestatus != null) {
			Eyestatus es = attrs.eyestatus;
			if (es.leftEyeOpen < eyeOpenMin || es.rightEyeOpen < eyeOpenMin) {
				return FaceAssessment.retry("눈이 너무 감겨 있습니다. 눈을 뜨고 다시 촬영해주세요", dto, rawJson, prettyJson, structure);
			}
		}

		return FaceAssessment.pass("적합한 사진입니다", dto, rawJson, prettyJson, structure);
	}

	private FaceStructure fetchAndComputeStructure(Face first, FaceAttributes attrs) throws Exception {
		String faceToken = first == null ? null : first.faceToken;
		if (faceToken == null || faceToken.isBlank()) {
			log.warn("Landmark 생략: face_token 이 비어 있습니다.");
			return null;
		}

		String tokenHint = faceToken.length() <= 10 ? faceToken : (faceToken.substring(0, 6) + "…" + faceToken.substring(faceToken.length() - 4));
		log.debug("Landmark API 호출 준비: face_token_hint={}, len={}", tokenHint, faceToken.length());

		String raw = postLandmarkRaw(faceToken, "106p");
		log.debug("Landmark API raw (106p), length={}:\n{}", raw == null ? -1 : raw.length(), summarizeRaw(raw));

		if (raw == null || raw.isBlank()) {
			log.warn("Landmark API 응답이 비어 있습니다.");
			return null;
		}

		JsonNode root = objectMapper.readTree(raw);
		JsonNode landmark = extractLandmark(root);
		String apiErr = faceppApiError(root);
		boolean usable = isNamedLandmarkUsable(landmark);

		if (apiErr != null || !usable) {
			if (apiErr != null) {
				log.warn("Landmark API(106p) 오류 응답: {}", apiErr);
			} else {
				log.warn("Landmark(106p) landmark 미사용: topKeys={}, lm_is_object={}, lm_size={}, numeric_key_heuristic={}",
						summarizeTopKeys(root),
						landmark.isObject(),
						landmark.isObject() ? landmark.size() : -1,
						landmark.isObject() && looksLikeNumericLandmarkKeys(landmark));
			}
			String raw2 = postLandmarkRaw(faceToken, "106");
			log.debug("Landmark API raw (106 fallback), length={}:\n{}", raw2 == null ? -1 : raw2.length(), summarizeRaw(raw2));
			if (raw2 == null || raw2.isBlank()) {
				return null;
			}
			root = objectMapper.readTree(raw2);
			apiErr = faceppApiError(root);
			if (apiErr != null) {
				log.warn("Landmark API(106) 오류 응답: {}", apiErr);
				return null;
			}
			landmark = extractLandmark(root);
		}

		if (!landmark.isObject()) {
			log.warn("Landmark JSON에서 landmark 객체를 찾지 못했습니다. topKeys={}", summarizeTopKeys(root));
			return null;
		}

		log.debug("Landmark 파싱: point_field_count={}, sample_keys={}", countLandmarkPoints(landmark), sampleLandmarkKeys(landmark));

		if (looksLikeNumericLandmarkKeys(landmark)) {
			log.warn("Landmark 키가 숫자 인덱스 위주입니다. 현재 코드는 이름 키(106p) 전제입니다. 샘플 키: {}", sampleLandmarkKeys(landmark));
			return null;
		}

		if (!isNamedLandmarkUsable(landmark)) {
			log.warn("Landmark에 필요한 이름 키가 부족합니다. size={}, keys_sample={}", landmark.size(), sampleLandmarkKeys(landmark));
			return null;
		}

		// 얼굴 가로/세로 비율: Detect의 face_rectangle (단순/요구사항 그대로)
		double faceW = first.faceRectangleWidth;
		double faceH = first.faceRectangleHeight;
		Double faceWH = (faceW > 0 && faceH > 0) ? (faceW / faceH) : null;

		Point2D leftEyeCenter = p(landmark, "left_eye_center");
		Point2D rightEyeCenter = p(landmark, "right_eye_center");
		Double eyeDistance = (leftEyeCenter != null && rightEyeCenter != null) ? dist(leftEyeCenter, rightEyeCenter) : null;

		// 눈 크기 비율(간단): 평균 눈 너비 / face_rectangle.width
		Double leftEyeWidth = eyeWidth(landmark, "left_eye_left_corner", "left_eye_right_corner");
		Double rightEyeWidth = eyeWidth(landmark, "right_eye_left_corner", "right_eye_right_corner");
		Double avgEyeWidth = avg(leftEyeWidth, rightEyeWidth);
		Double eyeWidthRatio = (avgEyeWidth != null && faceW > 0) ? (avgEyeWidth / faceW) : null;

		// 코-턱 vs 이마-코 거리 비율(근사):
		// nose_tip ~ contour_chin, (eyebrow upper middles 평균) ~ nose_tip
		Point2D noseTip = p(landmark, "nose_tip");
		Point2D chin = p(landmark, "contour_chin");
		Point2D lBrow = p(landmark, "left_eyebrow_upper_middle");
		Point2D rBrow = p(landmark, "right_eyebrow_upper_middle");
		Point2D browMid = (lBrow != null && rBrow != null) ? mid(lBrow, rBrow) : null;

		Double noseToChin = (noseTip != null && chin != null) ? dist(noseTip, chin) : null;
		Double browToNose = (browMid != null && noseTip != null) ? dist(browMid, noseTip) : null;
		Double verticalRatio = (noseToChin != null && browToNose != null && browToNose > 0) ? (noseToChin / browToNose) : null;

		// 입술 두께(근사): upper lip bottom ~ lower lip top
		Double lipThickness = lipThickness(landmark);

		// 앞머리(눈썹 가림) 추정: 눈썹 중심이 없거나, 눈썹-눈 거리(세로)가 비정상적으로 작으면 가려졌을 가능성
		boolean bangsLikely = estimateBangs(landmark, faceH);

		// 얼굴형 판정(휴리스틱): landmark의 폭/높이/이마·광대·턱 폭 조합
		String faceShape = classifyFaceShape(landmark, faceW, faceH, faceWH);

		FaceStructure s = new FaceStructure();
		s.eyeDistance = eyeDistance;
		s.faceWidthHeightRatio = faceWH;
		s.leftEyeWidthPx = leftEyeWidth;
		s.rightEyeWidthPx = rightEyeWidth;
		s.eyeWidthRatio = eyeWidthRatio;
		s.noseChinToForeheadNoseRatio = verticalRatio;
		s.lipThicknessPx = lipThickness;
		s.bangsLikely = bangsLikely;
		s.faceShape = faceShape;
		return s;
	}

	private String postLandmarkRaw(String faceToken, String landmarkType) {
		MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
		bodyBuilder.part("api_key", apiKey);
		bodyBuilder.part("api_secret", apiSecret);
		bodyBuilder.part("face_token", faceToken);
		bodyBuilder.part("landmark_type", landmarkType);

		MultiValueMap<String, org.springframework.http.HttpEntity<?>> multipart = bodyBuilder.build();
		String url = UriComponentsBuilder.fromUriString(LANDMARK_URL).build().toUriString();

		try {
			return webClient.post()
					.uri(url)
					.contentType(MediaType.MULTIPART_FORM_DATA)
					.body(BodyInserters.fromMultipartData(multipart))
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.block();
		} catch (WebClientResponseException e) {
			log.error("Landmark HTTP 실패 status={}, body={}", e.getStatusCode().value(), summarizeRaw(e.getResponseBodyAsString()));
			return null;
		}
	}

	private static String faceppApiError(JsonNode root) {
		if (root == null || root.isMissingNode()) return "empty json";
		JsonNode em = root.get("error_message");
		if (em != null && em.isTextual()) {
			String s = em.asText("");
			if (!s.isBlank()) return s;
		}
		return null;
	}

	private static String summarizeRaw(String raw) {
		if (raw == null) return "(null)";
		if (raw.length() <= 16000) return raw;
		return raw.substring(0, 16000) + "\n... (truncated, total_len=" + raw.length() + ")";
	}

	private static String summarizeTopKeys(JsonNode root) {
		if (root == null || !root.isObject()) return "(non-object)";
		StringBuilder sb = new StringBuilder();
		int n = 0;
		Iterator<String> it = root.fieldNames();
		while (it.hasNext() && n < 20) {
			if (n++ > 0) sb.append(", ");
			sb.append(it.next());
		}
		return sb.toString();
	}

	private static int countLandmarkPoints(JsonNode landmark) {
		if (landmark == null || !landmark.isObject()) return 0;
		int c = 0;
		Iterator<String> it = landmark.fieldNames();
		while (it.hasNext()) {
			it.next();
			c++;
		}
		return c;
	}

	private static String sampleLandmarkKeys(JsonNode landmark) {
		if (landmark == null || !landmark.isObject()) return "";
		StringBuilder sb = new StringBuilder();
		int n = 0;
		Iterator<String> it = landmark.fieldNames();
		while (it.hasNext() && n < 12) {
			if (n++ > 0) sb.append(", ");
			sb.append(it.next());
		}
		return sb.toString();
	}

	private static boolean looksLikeNumericLandmarkKeys(JsonNode landmark) {
		if (landmark == null || !landmark.isObject() || landmark.size() == 0) return false;
		int numeric = 0;
		int total = 0;
		Iterator<String> it = landmark.fieldNames();
		while (it.hasNext()) {
			String k = it.next();
			total++;
			boolean allDigit = !k.isEmpty();
			for (int i = 0; i < k.length(); i++) {
				if (!Character.isDigit(k.charAt(i))) {
					allDigit = false;
					break;
				}
			}
			if (allDigit) numeric++;
		}
		return total >= 40 && numeric >= (int) (total * 0.85);
	}

	/** 이름 기반 106p 랜드마크로 계산 가능한지 대략 판별 */
	private static boolean isNamedLandmarkUsable(JsonNode landmark) {
		if (!landmark.isObject()) return false;
		if (landmark.size() < 10) return false;
		if (looksLikeNumericLandmarkKeys(landmark)) return false;
		return landmark.has("left_eye_center")
				|| landmark.has("right_eye_center")
				|| landmark.has("nose_tip")
				|| landmark.has("contour_chin");
	}

	private static JsonNode extractLandmark(JsonNode root) {
		if (root == null) return MissingNode.getInstance();

		JsonNode direct = root.path("landmark");
		if (direct.isObject() && direct.size() > 0) return direct;

		// Common shapes (varies by API version / wrapper)
		JsonNode a = root.path("result").path("landmark");
		if (a.isObject() && a.size() > 0) return a;

		JsonNode facesA = root.path("result").path("faces");
		if (facesA.isArray() && facesA.size() > 0) {
			JsonNode lm = facesA.get(0).path("landmark");
			if (lm.isObject() && lm.size() > 0) return lm;
		}

		JsonNode facesB = root.path("faces");
		if (facesB.isArray() && facesB.size() > 0) {
			JsonNode lm = facesB.get(0).path("landmark");
			if (lm.isObject() && lm.size() > 0) return lm;
		}

		// result 자체가 landmark 맵인 경우
		JsonNode res = root.path("result");
		if (res.isObject() && res.has("left_eye_center")) {
			return res;
		}

		JsonNode alt1 = root.path("face_landmark");
		if (alt1.isObject() && alt1.size() > 0) return alt1;

		JsonNode alt2 = root.path("data").path("landmark");
		if (alt2.isObject() && alt2.size() > 0) return alt2;

		return MissingNode.getInstance();
	}

	private static Double eyeWidth(JsonNode landmark, String leftKey, String rightKey) {
		Point2D a = p(landmark, leftKey);
		Point2D b = p(landmark, rightKey);
		return (a != null && b != null) ? dist(a, b) : null;
	}

	private static Double lipThickness(JsonNode landmark) {
		// Face++ 106p 키는 버전에 따라 조금씩 달라질 수 있어 후보를 넓게 잡음
		Point2D upperBottom = firstPoint(landmark,
				"mouth_upper_lip_bottom",
				"mouth_upper_lip_lower_middle",
				"mouth_upper_lip_bottom_middle",
				"upper_lip_bottom",
				"upper_lip_lower_middle"
		);
		Point2D lowerTop = firstPoint(landmark,
				"mouth_lower_lip_top",
				"mouth_lower_lip_upper_middle",
				"mouth_lower_lip_top_middle",
				"lower_lip_top",
				"lower_lip_upper_middle"
		);
		if (upperBottom != null && lowerTop != null) {
			return dist(upperBottom, lowerTop);
		}
		return null;
	}

	private static boolean estimateBangs(JsonNode landmark, double faceH) {
		Point2D lBrow = p(landmark, "left_eyebrow_upper_middle");
		Point2D rBrow = p(landmark, "right_eyebrow_upper_middle");
		Point2D browMid = (lBrow != null && rBrow != null) ? mid(lBrow, rBrow) : null;
		if (browMid == null) return true;

		Point2D leftEyeCenter = p(landmark, "left_eye_center");
		Point2D rightEyeCenter = p(landmark, "right_eye_center");
		Point2D eyeMid = (leftEyeCenter != null && rightEyeCenter != null) ? mid(leftEyeCenter, rightEyeCenter) : null;
		if (eyeMid == null) return false;

		double dy = Math.abs(browMid.y - eyeMid.y);
		if (faceH <= 0) return dy < 10; // px fallback
		// 눈썹-눈 거리(세로)가 얼굴 높이 대비 너무 작으면 가려졌을 확률이 높음
		return (dy / faceH) < 0.06;
	}

	private static String classifyFaceShape(JsonNode landmark, double faceW, double faceH, Double faceWH) {
		// 폭 측정용 contour 포인트 후보(대략 이마/광대/턱 라인)
		Double cheekW = widthBetween(landmark, "contour_left5", "contour_right5");
		Double jawW = widthBetween(landmark, "contour_left9", "contour_right9");
		Double foreheadW = widthBetween(landmark, "left_eyebrow_left_corner", "right_eyebrow_right_corner");

		double wh = (faceWH != null) ? faceWH : ((faceW > 0 && faceH > 0) ? (faceW / faceH) : 0.0);

		// 기본 축: 길이감(wh가 낮을수록 길쭉)
		boolean longish = (wh > 0 && wh < 0.78);
		boolean wideish = (wh > 0.92);

		// 턱/광대/이마 폭 관계
		double cheek = cheekW == null ? 0.0 : cheekW;
		double jaw = jawW == null ? 0.0 : jawW;
		double fore = foreheadW == null ? 0.0 : foreheadW;

		boolean foreMuchWiderThanJaw = (fore > 0 && jaw > 0) && (fore / jaw > 1.12);
		boolean jawCloseToCheek = (jaw > 0 && cheek > 0) && (Math.abs(jaw - cheek) / cheek < 0.08);

		if (longish) return "Long";
		if (foreMuchWiderThanJaw) return "Heart";
		if (wideish && jawCloseToCheek) return "Square";
		if (wideish) return "Round";
		// 나머지: 타원형 우선
		return "Oval";
	}

	private static Double widthBetween(JsonNode landmark, String leftKey, String rightKey) {
		Point2D a = p(landmark, leftKey);
		Point2D b = p(landmark, rightKey);
		if (a == null || b == null) return null;
		return Math.abs(a.x - b.x);
	}

	private static Point2D firstPoint(JsonNode landmark, String... keys) {
		for (String k : keys) {
			Point2D p = p(landmark, k);
			if (p != null) return p;
		}
		return null;
	}

	private static Double avg(Double a, Double b) {
		if (a == null && b == null) return null;
		if (a == null) return b;
		if (b == null) return a;
		return (a + b) / 2.0;
	}

	private static Point2D p(JsonNode landmark, String key) {
		if (landmark == null || !landmark.isObject()) return null;
		JsonNode n = landmark.path(key);
		if (!n.isObject()) return null;
		double x = num(n, "x");
		double y = num(n, "y");
		return new Point2D(x, y);
	}

	private static Point2D mid(Point2D a, Point2D b) {
		return new Point2D((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
	}

	private static double dist(Point2D a, Point2D b) {
		double dx = a.x - b.x;
		double dy = a.y - b.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

	private static double bestOpenProb(JsonNode eyeStatusNode) {
		if (eyeStatusNode == null || eyeStatusNode.isMissingNode() || eyeStatusNode.isNull()) return 0.0;
		double noGlassOpen = num(eyeStatusNode, "no_glass_eye_open");
		double normalGlassOpen = num(eyeStatusNode, "normal_glass_eye_open");
		return Math.max(noGlassOpen, normalGlassOpen);
	}

	private static String text(JsonNode node, String field) {
		if (node == null) return "";
		return node.path(field).asText("");
	}

	private static double num(JsonNode node, String field) {
		if (node == null) return 0.0;
		JsonNode n = node.path(field);
		if (n.isNumber()) return n.asDouble();
		if (n.isTextual()) {
			try {
				return Double.parseDouble(n.asText());
			} catch (Exception ignored) {
				return 0.0;
			}
		}
		return 0.0;
	}

	private static double abs(double v) {
		return v < 0 ? -v : v;
	}

	public enum Verdict { PASS, FAIL, RETRY }

	public static class FaceAssessment {
		public Verdict verdict;
		public String reason;
		public int facesCount;
		public String firstFaceToken;
		public Headpose headpose;
		public Blur blur;
		public Eyestatus eyestatus;
		public Gender gender;
		public Age age;
		public String emotion;
		public String emotionSummary;
		public Beauty beauty;
		public Skinstatus skinstatus;
		public Facequality facequality;
		public Smile smile;
		public Glass glass;
		public FaceStructure structure;
		public String prettyJson;
		public String rawJson;

		public static FaceAssessment pass(String reason, DetectResponse dto, String rawJson, String prettyJson, FaceStructure structure) {
			return of(Verdict.PASS, reason, dto, rawJson, prettyJson, structure);
		}

		public static FaceAssessment fail(String reason, DetectResponse dto, String rawJson, String prettyJson, FaceStructure structure) {
			return of(Verdict.FAIL, reason, dto, rawJson, prettyJson, structure);
		}

		public static FaceAssessment retry(String reason, DetectResponse dto, String rawJson, String prettyJson, FaceStructure structure) {
			return of(Verdict.RETRY, reason, dto, rawJson, prettyJson, structure);
		}

		private static FaceAssessment of(Verdict verdict, String reason, DetectResponse dto, String rawJson, String prettyJson, FaceStructure structure) {
			FaceAssessment a = new FaceAssessment();
			a.verdict = verdict;
			a.reason = reason;
			a.facesCount = dto == null ? 0 : dto.faces.size();
			a.rawJson = rawJson;
			a.prettyJson = prettyJson;
			a.structure = structure;
			if (dto != null && !dto.faces.isEmpty()) {
				Face f = dto.faces.get(0);
				a.firstFaceToken = f.faceToken;
				if (f.attributes != null) {
					a.headpose = f.attributes.headpose;
					a.blur = f.attributes.blur;
					a.eyestatus = f.attributes.eyestatus;
					a.gender = f.attributes.gender;
					a.age = f.attributes.age;
					a.beauty = f.attributes.beauty;
					a.skinstatus = f.attributes.skinstatus;
					a.facequality = f.attributes.facequality;
					a.emotion = compactJson(f.attributes.emotionJson);
					a.emotionSummary = f.attributes.emotionSummary;
					a.smile = f.attributes.smile;
					a.glass = f.attributes.glass;
				}
			}
			return a;
		}
	}

	private static String compactJson(JsonNode node) {
		if (node == null || node.isMissingNode() || node.isNull()) return null;
		return node.toString();
	}

	// DTOs (최소 파싱용)
	public static class DetectResponse {
		public String requestId;
		public String imageId;
		public List<Face> faces = new ArrayList<>();
	}

	public static class Face {
		public String faceToken;
		public FaceAttributes attributes;
		public double faceRectangleWidth;
		public double faceRectangleHeight;
	}

	public static class FaceAttributes {
		public Headpose headpose;
		public Blur blur;
		public Eyestatus eyestatus;
		public Smile smile;
		public Glass glass;
		public Gender gender;
		public Age age;
		public JsonNode emotionJson;
		public String emotionSummary;
		public Beauty beauty;
		public Skinstatus skinstatus;
		public Facequality facequality;
	}

	public static class Headpose {
		public double yawAngle;
		public double pitchAngle;
		public double rollAngle;
	}

	public static class Blur {
		public double blurnessValue;
		public double blurnessThreshold;
		public double motionblurValue;
		public double motionblurThreshold;
		public double gaussianblurValue;
		public double gaussianblurThreshold;
	}

	public static class Eyestatus {
		public double leftEyeOpen;
		public double rightEyeOpen;
	}

	public static class Smile {
		public double value;
		public double threshold;
	}

	public static class Glass {
		public String value;
	}

	public static class Gender {
		public String value;
		public double confidence;
	}

	public static class Age {
		public double value;
	}

	public static class Beauty {
		public double maleScore;
		public double femaleScore;
	}

	public static class Skinstatus {
		public double health;
		public double stain;
		public double acne;
		public double darkCircle;
	}

	public static class Facequality {
		public double value;
		public double threshold;
	}

	private static String summarizeEmotion(JsonNode emotion) {
		if (emotion == null || emotion.isMissingNode() || !emotion.isObject()) return null;

		String top1Name = null;
		double top1Val = -1;
		String top2Name = null;
		double top2Val = -1;

		var it = emotion.fields();
		while (it.hasNext()) {
			var e = it.next();
			double v = e.getValue().isNumber() ? e.getValue().asDouble() : 0.0;
			String name = e.getKey();
			if (v > top1Val) {
				top2Val = top1Val; top2Name = top1Name;
				top1Val = v; top1Name = name;
			} else if (v > top2Val) {
				top2Val = v; top2Name = name;
			}
		}

		if (top1Name == null) return null;
		if (top2Name == null) return "%s %.1f%%".formatted(top1Name, top1Val);
		return "%s %.1f%%, %s %.1f%%".formatted(top1Name, top1Val, top2Name, top2Val);
	}

	public static class FaceStructure {
		public Double eyeDistance;
		public Double faceWidthHeightRatio;
		public Double leftEyeWidthPx;
		public Double rightEyeWidthPx;
		public Double eyeWidthRatio;
		public Double noseChinToForeheadNoseRatio;
		public Double lipThicknessPx;
		public String faceShape; // Round / Oval / Square / Long / Heart
		public boolean bangsLikely;
	}

	private record Point2D(double x, double y) {}
}

