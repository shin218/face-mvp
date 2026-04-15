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
public class OpenAiVibeService {

	private static final String CHAT_URL = "https://api.openai.com/v1/chat/completions";
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(60);

	private final WebClient webClient;
	private final ObjectMapper objectMapper;
	private final String apiKey;
	private final String model;

	public OpenAiVibeService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${openai.api.key:}") String apiKey,
			@Value("${openai.api.model:gpt-4o}") String model
	) {
		this.objectMapper = objectMapper;
		this.apiKey = apiKey == null ? "" : apiKey.trim();
		this.model = model;

		HttpClient httpClient = HttpClient.create().responseTimeout(RESPONSE_TIMEOUT);
		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}

	public boolean isConfigured() {
		return apiKey != null && !apiKey.isBlank();
	}

	public String analyzeVibe(VibeInput input) {
		if (!isConfigured()) {
			throw new IllegalStateException("OPENAI_API_KEY(또는 openai.api.key)가 설정되지 않았습니다.");
		}
		try {
			ObjectNode body = objectMapper.createObjectNode();
			body.put("model", model);
			body.put("temperature", 0.55);
			body.put("max_tokens", 900);

			ArrayNode messages = body.putArray("messages");
			ObjectNode systemMsg = messages.addObject();
			systemMsg.put("role", "system");
			systemMsg.put("content", VibePrompts.SYSTEM_PROMPT);
			ObjectNode userMsg = messages.addObject();
			userMsg.put("role", "user");
			userMsg.put("content", VibePrompts.buildUserText(input));

			String raw = webClient.post()
					.uri(CHAT_URL)
					.header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
					.header(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
					.contentType(MediaType.APPLICATION_JSON)
					.bodyValue(body)
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.onErrorMap(TimeoutException.class, ex ->
							new RuntimeException("OpenAI API 응답이 지연되어 타임아웃이 발생했습니다. (최대 " + RESPONSE_TIMEOUT.toSeconds() + "초)", ex)
					)
					.block();

			if (raw == null || raw.isBlank()) {
				throw new RuntimeException("OpenAI API 응답이 비어있습니다.");
			}
			return extractAssistantText(raw);
		} catch (WebClientResponseException e) {
			String resp = e.getResponseBodyAsString();
			String msg = "OpenAI API 호출 실패 (" + e.getStatusCode().value() + ")";
			if (resp != null && !resp.isBlank()) {
				msg += ": " + resp;
			}
			throw new RuntimeException(msg, e);
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage() == null ? "OpenAI API 호출 중 오류가 발생했습니다." : e.getMessage(), e);
		}
	}

	private String extractAssistantText(String rawJson) throws Exception {
		JsonNode root = objectMapper.readTree(rawJson);
		JsonNode choices = root.path("choices");
		if (choices.isArray() && choices.size() > 0) {
			String text = choices.get(0).path("message").path("content").asText("");
			if (!text.isBlank()) {
				return text.trim();
			}
		}
		throw new RuntimeException("OpenAI 응답에서 assistant 텍스트를 찾지 못했습니다.");
	}
}
