package com.example.demo.service;

import com.example.demo.service.vibe.VibeInput;
import com.example.demo.service.vibe.VibePrompts;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;
import java.util.concurrent.TimeoutException;

@Service
public class ClaudeVibeService {

	private static final String CLAUDE_MESSAGES_URL = "https://api.anthropic.com/v1/messages";
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(60);

	private final WebClient webClient;
	private final ObjectMapper objectMapper;

	private final String apiKey;
	private final String model;

	public ClaudeVibeService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${claude.api.key:}") String apiKey,
			@Value("${claude.api.model:claude-sonnet-4-6}") String model
	) {
		this.objectMapper = objectMapper;
		this.apiKey = apiKey == null ? "" : apiKey.trim();
		this.model = model;

		HttpClient httpClient = HttpClient.create()
				.responseTimeout(RESPONSE_TIMEOUT);

		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}

	public boolean isConfigured() {
		return apiKey != null && !apiKey.isBlank();
	}

	public String analyzeVibe(VibeInput input) {
		if (!isConfigured()) {
			throw new IllegalStateException("ANTHROPIC_API_KEY(또는 claude.api.key)가 설정되지 않았습니다.");
		}
		try {
			ObjectNode body = buildRequestBody(input);

			String raw = webClient.post()
					.uri(CLAUDE_MESSAGES_URL)
					.header("x-api-key", apiKey)
					.header("anthropic-version", "2023-06-01")
					.header(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
					.contentType(MediaType.APPLICATION_JSON)
					.bodyValue(body)
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.onErrorMap(TimeoutException.class, ex ->
							new RuntimeException("Claude API 응답이 지연되어 타임아웃이 발생했습니다. (최대 " + RESPONSE_TIMEOUT.toSeconds() + "초)", ex)
					)
					.block();

			if (raw == null || raw.isBlank()) {
				throw new RuntimeException("Claude API 응답이 비어있습니다.");
			}

			return extractTextFromResponse(raw);
		} catch (WebClientResponseException e) {
			String resp = e.getResponseBodyAsString();
			String msg = "Claude API 호출 실패 (" + e.getStatusCode().value() + ")";
			if (resp != null && !resp.isBlank()) {
				msg += ": " + resp;
			}
			throw new RuntimeException(msg, e);
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage() == null ? "Claude API 호출 중 오류가 발생했습니다." : e.getMessage(), e);
		}
	}

	private ObjectNode buildRequestBody(VibeInput input) {
		ObjectNode root = objectMapper.createObjectNode();
		root.put("model", model);
		root.put("max_tokens", 4096);
		root.put("temperature", 0.55);

		root.put("system", VibePrompts.SYSTEM_PROMPT);

		ArrayNode messages = root.putArray("messages");
		ObjectNode userMsg = messages.addObject();
		userMsg.put("role", "user");
		ArrayNode content = userMsg.putArray("content");
		ObjectNode text = content.addObject();
		text.put("type", "text");
		text.put("text", VibePrompts.buildUserText(input));

		return root;
	}

	private String extractTextFromResponse(String rawJson) throws Exception {
		JsonNode root = objectMapper.readTree(rawJson);
		JsonNode content = root.path("content");
		if (content.isArray() && content.size() > 0) {
			StringBuilder sb = new StringBuilder();
			for (JsonNode block : content) {
				if (!block.has("text")) {
					continue;
				}
				String t = block.get("text").asText("");
				if (t.isBlank()) {
					continue;
				}
				if (sb.length() > 0) {
					sb.append('\n');
				}
				sb.append(t);
			}
			String out = sb.toString().trim();
			if (!out.isBlank()) {
				return out;
			}
		}
		throw new RuntimeException("Claude 응답에서 텍스트를 찾지 못했습니다.");
	}
}
