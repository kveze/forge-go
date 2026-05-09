package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ==================== СТРУКТУРЫ ====================

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenRouterRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type OpenRouterResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type WorkoutRequest struct {
	Days      int    `json:"days"`
	Gender    string `json:"gender"`
	Age       int    `json:"age"`
	Height    int    `json:"height"`
	Weight    int    `json:"weight"`
	Goal      string `json:"goal"`
	Level     string `json:"level"`
	Equipment string `json:"equipment"`
}

type Exercise struct {
	Name    string `json:"name"`
	Sets    int    `json:"sets"`
	Reps    string `json:"reps"`
	RestSec int    `json:"rest_sec"`
	Notes   string `json:"notes,omitempty"`
}

type DayPlan struct {
	Day       int        `json:"day"`
	Focus     string     `json:"focus"`
	Exercises []Exercise `json:"exercises"`
}

type WorkoutPlan struct {
	WeekPlan  []DayPlan `json:"week_plan"`
	Goal      string    `json:"goal"`
	Level     string    `json:"level"`
	CreatedAt string    `json:"created_at"`
}

type TipsRequest struct {
	Gender string `json:"gender"`
	Age    int    `json:"age"`
	Weight int    `json:"weight"`
	Goal   string `json:"goal"`
	Level  string `json:"level"`
	Plan   string `json:"plan"`
}

type TipsResponse struct {
	Tips []string `json:"tips"`
}

type RecoveryRequest struct {
	Age   int    `json:"age"`
	Goal  string `json:"goal"`
	Level string `json:"level"`
}

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type LooksMaxTipRequest struct {
	Age    int    `json:"age"`
	Gender string `json:"gender"`
	Weight int    `json:"weight"`
	Height int    `json:"height"`
	Goal   string `json:"goal"`
	Level  string `json:"level"`
	Days   int    `json:"days"`
}

type LooksMaxTipResponse struct {
	Tips     []string `json:"tips"`
	Priority string   `json:"priority"`
	Category string   `json:"category"`
}

type LooksMaxAnalyzeRequest struct {
	ImageBase64 string `json:"image_base64"`
	Age         int    `json:"age"`
	Gender      string `json:"gender"`
	Goal        string `json:"goal"`
}

type LooksMaxAnalyzeResponse struct {
	Tips     []string `json:"tips"`
	Priority string   `json:"priority"`
	Category string   `json:"category"`
}

type LooksMaxTransformRequest struct {
	ImageBase64 string   `json:"image_base64"`
	Tips        []string `json:"tips"`
	Gender      string   `json:"gender"`
}

type LooksMaxTransformResponse struct {
	ImageBase64 string `json:"image_base64"`
}


type ChatRequest struct {
    Messages []Message `json:"messages"`
    UserData map[string]interface{} `json:"userData"`
    Plan     interface{} `json:"plan"`
}

type ChatAPIRequest struct {
    Model    string    `json:"model"`
    Messages []Message `json:"messages"`
}
// ==================== RATE LIMITER ====================

type RateLimiter struct {
	mu       sync.Mutex
	requests map[string][]time.Time
	limit    int
	window   time.Duration
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		requests: make(map[string][]time.Time),
		limit:    limit,
		window:   window,
	}
}

func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-rl.window)

	var valid []time.Time
	for _, t := range rl.requests[ip] {
		if t.After(windowStart) {
			valid = append(valid, t)
		}
	}
	rl.requests[ip] = valid

	if len(rl.requests[ip]) >= rl.limit {
		return false
	}

	rl.requests[ip] = append(rl.requests[ip], now)
	return true
}

// ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

var (
	httpClient  *http.Client
	rateLimiter *RateLimiter
	apiKey      string
	openaiKey   string
)

// ==================== MIDDLEWARE ====================

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		allowedOrigins := []string{
			"https://forge-client-main.netlify.app",
			"http://localhost:5173",
			"http://localhost:5174",
			"http://localhost:3000",
		}

		allowed := false
		for _, o := range allowedOrigins {
			if o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}

		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next(w, r)
	}
}

func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		clientKey := r.Header.Get("X-API-Key")
		if clientKey != "" && clientKey != os.Getenv("CLIENT_API_KEY") {
			sendError(w, "Неверный API ключ", http.StatusUnauthorized)
			return
		}

		ip := r.RemoteAddr
		if !rateLimiter.Allow(ip) {
			sendError(w, "Слишком много запросов. Попробуйте позже.", http.StatusTooManyRequests)
			return
		}

		next(w, r)
	}
}


func recoverMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("PANIC: %v", err)
				sendError(w, "Внутренняя ошибка сервера", http.StatusInternalServerError)
			}
		}()
		next(w, r)
	}
}


// ==================== УТИЛИТЫ ====================

func sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func sendError(w http.ResponseWriter, message string, status int) {
	sendJSON(w, APIResponse{Success: false, Error: message}, status)
}

func getClientIP(r *http.Request) string {
	ip := r.Header.Get("X-Forwarded-For")
	if ip != "" {
		return strings.Split(ip, ",")[0]
	}
	return r.RemoteAddr
}

// ==================== AI ЗАПРОС ====================

func askAI(prompt string, systemPrompt string) (string, error) {
	if apiKey == "" {
		return "", fmt.Errorf("API ключ не настроен")
	}

	reqBody := OpenRouterRequest{
		Model: "openai/gpt-4o-mini",
		Messages: []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: prompt},
		},
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("ошибка маршалинга: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("ошибка создания запроса: %w", err)
	}

	req.Header.Set("Authorization", "Bearer " + apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://forge-client-main.netlify.app")
	req.Header.Set("X-Title", "AI Fitness Trainer")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ошибка запроса к AI: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("AI вернул ошибку %d: %s", resp.StatusCode, string(body))
	}

	var result OpenRouterResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("ошибка парсинга ответа: %w", err)
	}

	if result.Error != nil {
		return "", fmt.Errorf("AI ошибка: %s", result.Error.Message)
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("пустой ответ от AI")
	}

	return result.Choices[0].Message.Content, nil
}

// ==================== ВАЛИДАЦИЯ ====================

func validateWorkoutRequest(req WorkoutRequest) error {
	if req.Days < 2 || req.Days > 5 {
		return fmt.Errorf("дни: от 2 до 5")
	}
	if req.Age < 10 || req.Age > 100 {
		return fmt.Errorf("возраст: от 10 до 100")
	}
	if req.Height < 100 || req.Height > 250 {
		return fmt.Errorf("рост: от 100 до 250 см")
	}
	if req.Weight < 30 || req.Weight > 300 {
		return fmt.Errorf("вес: от 30 до 300 кг")
	}

	goal := strings.ToLower(req.Goal)
	validGoals := []string{"набор массы", "сила", "рельеф", "выносливость", "похудение"}
	valid := false
	for _, g := range validGoals {
		if goal == g {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("неверная цель: %s", req.Goal)
	}

	level := strings.ToLower(req.Level)
	validLevels := []string{"новичок", "средний", "продвинутый"}
	valid = false
	for _, l := range validLevels {
		if level == l {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("неверный уровень: %s", req.Level)
	}

	return nil
}

// ==================== HANDLERS ====================

func generateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req WorkoutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных: "+err.Error(), http.StatusBadRequest)
		return
	}

	if err := validateWorkoutRequest(req); err != nil {
		sendError(w, "Ошибка валидации: "+err.Error(), http.StatusBadRequest)
		return
	}

	systemPrompt := `Ты профессиональный персональный фитнес-тренер(немного тайлер дерден и девид гоггинс) с 15-летним опытом.
Твоя задача — создавать безопасные и эффективные планы тренировок.

ПРАВИЛА:
1. Используй ТОЛЬКО реальные упражнения (никаких вымышленных "силовых потоков" или "энергетических ударов")
2. НЕ придумывай несуществующие упражнения - если не хватает оборудования — замени на похожее или предложи вариант без оборудования
3. Учитывай уровень подготовки и доступное оборудование 
4. НЕ нагружай одну группу мышц два дня подряд 
5. Верни ТОЛЬКО валидный JSON без markdown, без пояснений и вступлений. Никаких других символов и языков, кроме кириллицы и латиницы. Никаких иероглифов, специальных символов, unicode мусора.

ФОРМАТ ОТВЕТА (строго):
{
  "week_plan": [
    {
      "day": 1,
      "focus": "Грудь/Трицепс",
      "exercises": [
        {"name": "Отжимания", "sets": 3, "reps": "12-15", "rest_sec": 60, "notes": "Держи спину ровно"}
      ]
    }
  ],
  "goal": "цель",
  "level": "уровень",
  "created_at": "2025-01-01"
}`

	prompt := fmt.Sprintf(`Создай план тренировок на %d дней в неделю.

ДАННЫЕ КЛИЕНТА:
- Пол: %s
- Возраст: %d лет
- Рост: %d см
- Вес: %d кг
- Цель: %s
- Уровень: %s
- Оборудование: %s

ВАЖНО:
- Новичок: 3 подхода, 12-15 повторений, отдых 60-90с
- Средний: 4 подхода, 8-12 повторений, отдых 60с
- Продвинутый: 4-5 подходов, 5-10 повторений, отдых 45-60с

Верни ТОЛЬКО JSON.`,
		req.Days, req.Gender, req.Age, req.Height, req.Weight,
		req.Goal, req.Level, req.Equipment)

	content, err := askAI(prompt, systemPrompt)
	if err != nil {
		log.Printf("AI Error: %v", err)
		sendError(w, "Ошибка генерации плана: "+err.Error(), http.StatusInternalServerError)
		return
	}

	var plan WorkoutPlan
	if err := json.Unmarshal([]byte(content), &plan); err != nil {
		cleaned := strings.TrimSpace(content)
		cleaned = strings.TrimPrefix(cleaned, "```json")
		cleaned = strings.TrimPrefix(cleaned, "```")
		cleaned = strings.TrimSuffix(cleaned, "```")

		if err := json.Unmarshal([]byte(cleaned), &plan); err != nil {
			log.Printf("JSON Parse Error: %v, Content: %s", err, content)
			sendError(w, "Ошибка формата ответа AI", http.StatusInternalServerError)
			return
		}
	}

	plan.Goal = req.Goal
	plan.Level = req.Level
	plan.CreatedAt = time.Now().Format(time.RFC3339)

	sendJSON(w, APIResponse{Success: true, Data: plan}, http.StatusOK)
}

func tipsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req TipsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных", http.StatusBadRequest)
		return
	}

	systemPrompt := `Ты нутрициолог. Давай конкретные советы по питанию.
Верни ТОЛЬКО JSON массив строк. Без markdown. Без вступления.

ФОРМАТ: {"tips": ["совет 1", "совет 2", ...]}`

	prompt := fmt.Sprintf(`Дай 6 конкретных советов по питанию.

ДАННЫЕ:
- Пол: %s
- Возраст: %d лет
- Вес: %d кг
- Цель: %s
- Уровень: %s

План тренировок: %s`,
		req.Gender, req.Age, req.Weight, req.Goal, req.Level, req.Plan)

	content, err := askAI(prompt, systemPrompt)
	if err != nil {
		sendError(w, "Ошибка генерации советов: "+err.Error(), http.StatusInternalServerError)
		return
	}

	var response TipsResponse
	if err := json.Unmarshal([]byte(content), &response); err != nil {
		sendError(w, "Ошибка формата ответа", http.StatusInternalServerError)
		return
	}

	sendJSON(w, APIResponse{Success: true, Data: response}, http.StatusOK)
}

func recoveryHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req RecoveryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных", http.StatusBadRequest)
		return
	}

	systemPrompt := `Ты эксперт по восстановлению. Давай конкретные советы.
Верни ТОЛЬКО JSON массив строк. Без markdown.

ФОРМАТ: {"tips": ["совет 1", "совет 2", ...]}`

	prompt := fmt.Sprintf(`Дай 5 конкретных советов по сну и восстановлению.

ДАННЫЕ:
- Возраст: %d лет
- Цель: %s
- Уровень: %s`,
		req.Age, req.Goal, req.Level)

	content, err := askAI(prompt, systemPrompt)
	if err != nil {
		sendError(w, "Ошибка генерации советов: "+err.Error(), http.StatusInternalServerError)
		return
	}

	var response TipsResponse
	if err := json.Unmarshal([]byte(content), &response); err != nil {
		sendError(w, "Ошибка формата ответа", http.StatusInternalServerError)
		return
	}

	sendJSON(w, APIResponse{Success: true, Data: response}, http.StatusOK)
}



func healthHandler(w http.ResponseWriter, r *http.Request) {
	sendJSON(w, APIResponse{Success: true, Data: map[string]string{
		"status":    "ok",
		"timestamp": time.Now().Format(time.RFC3339),
	}}, http.StatusOK)
}


func chatHandler(w http.ResponseWriter, r *http.Request) {
    var body ChatRequest
    json.NewDecoder(r.Body).Decode(&body)

	log.Printf("Chat request - userData: %v, plan: %v", body.UserData, body.Plan)

    system := buildSystemPrompt(body.UserData, body.Plan)

    allMessages := []Message{{Role: "system", Content: system}}
	allMessages = append(allMessages, body.Messages...)

    reqBody := ChatAPIRequest{
        Model: "openai/gpt-4o-mini",
        Messages: allMessages,
    }

    jsonBody, _ := json.Marshal(reqBody)

    req, _ := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewBuffer(jsonBody))
    req.Header.Set("Authorization", "Bearer "+apiKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("HTTP-Referer", "https://forge-client-main.netlify.app")

    resp, err := httpClient.Do(req)
    if err != nil {
        w.WriteHeader(500)
        json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
        return
    }
    defer resp.Body.Close()

    body2, _ := io.ReadAll(resp.Body)

    // Парсим ответ чтобы извлечь план если есть
    var openRouterResp OpenRouterResponse
    if err := json.Unmarshal(body2, &openRouterResp); err == nil && len(openRouterResp.Choices) > 0 {
        content := openRouterResp.Choices[0].Message.Content

        // Ищем блок с планом
        planStart := strings.Index(content, "ПЛАН_СТАРТ")
        planEnd := strings.Index(content, "ПЛАН_КОНЕЦ")

        if planStart != -1 && planEnd != -1 {
            planJSON := strings.TrimSpace(content[planStart+len("ПЛАН_СТАРТ") : planEnd])
            textAfter := strings.TrimSpace(content[planEnd+len("ПЛАН_КОНЕЦ"):])

            var plan WorkoutPlan
            if err := json.Unmarshal([]byte(planJSON), &plan); err == nil {
                // Возвращаем специальный ответ с планом
                w.Header().Set("Content-Type", "application/json")
                json.NewEncoder(w).Encode(map[string]interface{}{
                    "choices": []map[string]interface{}{
                        {
                            "message": map[string]interface{}{
                                "content": textAfter,
                                "role":    "assistant",
                            },
                        },
                    },
                    "plan": plan,
                })
                return
            }
        }
    }

    // Обычный ответ без плана
    w.Header().Set("Content-Type", "application/json")
    w.Write(body2)
}

func buildSystemPrompt(userData map[string]interface{}, plan interface{}) string {
    planStr := ""
    if plan != nil {
        planBytes, _ := json.Marshal(plan)
        planStr = string(planBytes)
    }

    return fmt.Sprintf(`Ты персональный AI тренер FORGE. Строгий, немного тайлер дерден и девидгогинс, честный, поддерживающий. Говоришь кратко. - - Если пользователь говорит что хочет изменить цель или программу — помоги и сгенерируй новый план
- Если вопрос совсем не связан с фитнесом, здоровьем или телом — вежливо верни к теме

%s
%s

ПРАВИЛА ОБЩЕНИЯ:
- Отвечай коротко — 2-4 предложения
- Без markdown и звёздочек
- Пиши ТОЛЬКО на русском языке, никаких других символов и языков
- Используй ТОЛЬКО кириллицу и латиницу. Никаких иероглифов, специальных символов, unicode мусора.

ГЕНЕРАЦИЯ ПЛАНА:
Если ты собрал все данные пользователя (пол, возраст, вес, цель, оборудование, количество дней) — 
сгенерируй план тренировок и верни его в специальном формате:

ПЛАН_СТАРТ
{"week_plan": [{"day": 1, "focus": "...", "exercises": [{"name": "...", "sets": 3, "reps": "10-12", "rest_sec": 60, "notes": "..."}]}], "goal": "...", "level": "..."}
ПЛАН_КОНЕЦ



ИЗМЕНЕНИЕ ПЛАНА — ОБЯЗАТЕЛЬНО:
Если пользователь говорит "слишком легко", "слишком тяжело", "упрости", "усложни" или что-то похожее — 
ты ОБЯЗАН ответить новым планом. БЕЗ ИСКЛЮЧЕНИЙ.
Алгоритм:
1. Напиши 1 предложение что именно изменил
2. Сразу верни ПОЛНЫЙ новый план в формате ПЛАН_СТАРТ...ПЛАН_КОНЕЦ
3. Если план слишком лёгкий — увеличь подходы, повторения или добавь упражнения
4. Если слишком тяжёлый — уменьши подходы, повторения или убери упражнения
НЕ ДАВАЙ СОВЕТЫ СЛОВАМИ — ТОЛЬКО НОВЫЙ ПЛАН.

После блока с планом добавь короткий комментарий тренера.

Если данных недостаточно — спроси недостающие. Не генерируй план без всех данных.`,
        func() string {
            if userData != nil {
                bytes, _ := json.Marshal(userData)
                return "ДАННЫЕ КЛИЕНТА: " + string(bytes)
            }
            return "Данные клиента неизвестны — собери их в разговоре."
        }(),
        func() string {
            if planStr != "" {
                return "ТЕКУЩИЙ ПЛАН: " + planStr
            }
            return ""
        }(),
    )
}

// ==================== LOOKSMAX TIP ====================

func looksMaxTipHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req LooksMaxTipRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных", http.StatusBadRequest)
		return
	}

	if req.Age < 10 || req.Age > 100 {
		sendError(w, "Некорректный возраст", http.StatusBadRequest)
		return
	}
	if req.Height < 100 || req.Height > 250 {
		sendError(w, "Некорректный рост", http.StatusBadRequest)
		return
	}
	if req.Weight < 30 || req.Weight > 300 {
		sendError(w, "Некорректный вес", http.StatusBadRequest)
		return
	}

	if req.Age < 18 {
		allTips := []string{
			"База которую все игнорируют: умывайся утром и вечером. Просто вода и пенка — кожа уже другая через неделю.",
			"Самое важное что никто не делает: не дави прыщи. Серьёзно. Один выдавленный прыщ = шрам на месяцы.",
			"Чинтакс делай и вообщем шею качай",
			"Умывайся 2 раза в день и не трогай лицо руками. Руки переносят бактерии — это буквально причина половины прыщей.",
			"Bro, looksmax это не твоё пока. Сейчас одна задача — вырасти. Ложись до 23:00, вставай в 7-8.",
			"Слушай, heightmax сейчас реально важнее всего. Не сутулься — ты буквально теряешь 3-5 см просто так.",
			"Про лицо успеешь, не спеши. Белок 2г/кг каждый день — строительный материал для костей.",
			"Looksmax подождёт. Вредностей меньше — резкий инсулин глушит соматотропин.",
			"Ты ещё растёшь, это главное преимущество. Цинк + магний перед сном.",
			"Heightmax mode, всё остальное потом. Молоко каждый день — IGF-1 стимулирует рост костей.",
			"Спи в тёмной комнате и без телефона за 30 минут до сна — мелатонин запускает выброс гормона роста.",
			"Не пей много воды перед сном — нарушает качество восстановления.",
			"Солнце утром 15-20 минут — запускает метаболизм и гормон роста.",
			"Не пропускай завтрак — омлет или овсянка с фруктами.",
			"Избегай стрессов — хронический стресс повышает кортизол, подавляет рост.",
			"Питайся разнообразно — витамины и минералы из разных продуктов.",
			"Не сравнивай себя с другими — каждый растёт в своём темпе.",
		}

		n := int64(len(allTips))
		idx1 := time.Now().UnixNano() % n
		idx2 := (idx1 + 1 + time.Now().UnixNano()%(n-1)) % n

		sendJSON(w, APIResponse{Success: true, Data: LooksMaxTipResponse{
			Tips:     []string{allTips[idx1], allTips[idx2]},
			Priority: "высокий",
			Category: "тело",
		}}, http.StatusOK)
		return
	}

	bmi := float64(req.Weight) / math.Pow(float64(req.Height)/100, 2)
	bmiStr := fmt.Sprintf("%.1f", bmi)

	systemPrompt := `Ты научный looksmaxxer с реальным опытом. Не коуч, не мотиватор — практик.

ТВОЙ СТИЛЬ СОВЕТОВ (примеры как надо):
- "Снизь BF до 15%. Дефицит 500 ккал + 10к шагов ежедневно."
- "Умывайся 2x в день + увлажняй. Чистая кожа = +50 к внешке."
- "Ретинол на ночь + вода 2л. Кожа начнёт меняться за 4 недели."
- "Убери сахар и фастфуд на 30 дней — лицо похудеет визуально."

ПРАВИЛА:
- 1-2 предложения МАКСИМУМ
- Только конкретика: цифры, продукты, действия, сроки
- Никакой воды и мотивации
- Тон: прямой, как друг который реально шарит
- Высокий BMI (25+) → тело первично
- Нормальный BMI → лицо/уход/стиль

Верни ТОЛЬКО JSON без markdown:
{"tip": "...", "priority": "высокий/средний/низкий", "category": "лицо/тело/стиль/здоровье/уход"}`

	prompt := fmt.Sprintf(`Дай один лучший looksmax совет для этого человека.

ДАННЫЕ:
- Пол: %s
- Возраст: %d лет
- Рост: %d см
- Вес: %d кг
- BMI: %s
- Фитнес-цель: %s
- Уровень подготовки: %s
- Тренировок в неделю: %d (план УЖЕ составлен, не советуй менять количество тренировок)

ЗАПРЕЩЕНО: советовать что-либо про тренировки — план уже есть.

Верни ТОЛЬКО JSON.`,
		req.Gender, req.Age, req.Height, req.Weight, bmiStr, req.Goal, req.Level, req.Days)

	content, err := askAI(prompt, systemPrompt)
	if err != nil {
		log.Printf("LooksMax AI Error: %v", err)
		sendError(w, "Ошибка генерации совета: "+err.Error(), http.StatusInternalServerError)
		return
	}

	cleaned := strings.TrimSpace(content)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	cleaned = strings.TrimSpace(cleaned)

	var aiResp struct {
		Tip      string `json:"tip"`
		Priority string `json:"priority"`
		Category string `json:"category"`
	}
	if err := json.Unmarshal([]byte(cleaned), &aiResp); err != nil {
		log.Printf("LooksMax JSON Parse Error: %v, Content: %s", err, content)
		sendError(w, "Ошибка формата ответа AI", http.StatusInternalServerError)
		return
	}

	sendJSON(w, APIResponse{Success: true, Data: LooksMaxTipResponse{
		Tips:     []string{aiResp.Tip},
		Priority: aiResp.Priority,
		Category: aiResp.Category,
	}}, http.StatusOK)
}

// ==================== LOOKSMAX ANALYZE ====================

func looksMaxAnalyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req LooksMaxAnalyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных", http.StatusBadRequest)
		return
	}

	if req.ImageBase64 == "" {
		sendError(w, "Нет фото", http.StatusBadRequest)
		return
	}

	log.Printf("LooksMax Analyze: age=%d gender=%s imageLen=%d", req.Age, req.Gender, len(req.ImageBase64))

	if len(req.ImageBase64) > 7*1024*1024 {
		sendError(w, "Фото слишком большое", http.StatusBadRequest)
		return
	}

	imgBytes, err := base64.StdEncoding.DecodeString(req.ImageBase64)
	if err != nil {
		sendError(w, "Некорректное фото", http.StatusBadRequest)
		return
	}

	mimeType := "image/jpeg"
	isJPEG := len(imgBytes) > 2 && imgBytes[0] == 0xFF && imgBytes[1] == 0xD8
	isPNG := len(imgBytes) > 4 && string(imgBytes[1:4]) == "PNG"
	isWEBP := len(imgBytes) > 12 && string(imgBytes[8:12]) == "WEBP"

	if !isJPEG && !isPNG && !isWEBP {
		sendError(w, "Поддерживаются только JPEG, PNG, WEBP", http.StatusBadRequest)
		return
	}
	if isPNG {
		mimeType = "image/png"
	} else if isWEBP {
		mimeType = "image/webp"
	}

	systemText := `Ты научный looksmaxxer с реальным опытом. Сначала проверь — есть ли на фото лицо человека.
Анализируй лицо на фото и давай конкретные советы.
Стиль: прямой, как друг который реально шарит. Без воды.
Если лица нет — верни: {"error": "no_face"}
Если лицо есть — дай 3 конкретных looksmax совета.
Верни ТОЛЬКО JSON без markdown: {"tips": ["совет 1", "совет 2", "совет 3"], "priority": "высокий", "category": "лицо"}`

	openRouterKey := os.Getenv("OPENROUTER_KEY")

	requestBody := map[string]interface{}{
		"model": "meta-llama/llama-4-maverick:free",
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type": "text",
						"text": fmt.Sprintf("%s\n\nВозраст: %d, пол: %s, цель: %s. Дай 3 конкретных looksmax совета по этому лицу.", systemText, req.Age, req.Gender, req.Goal),
					},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:%s;base64,%s", mimeType, req.ImageBase64),
						},
					},
				},
			},
		},
	}

	jsonBody, _ := json.Marshal(requestBody)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewBuffer(jsonBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+openRouterKey)

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		sendError(w, "Ошибка запроса: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	log.Printf("OpenRouter status: %d, body: %s", resp.StatusCode, string(respBody))

	var orResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	json.Unmarshal(respBody, &orResp)

	if len(orResp.Choices) == 0 {
		sendError(w, "Нет ответа от модели", http.StatusInternalServerError)
		return
	}

	rawText := strings.TrimSpace(orResp.Choices[0].Message.Content)
	rawText = strings.TrimPrefix(rawText, "```json")
	rawText = strings.TrimPrefix(rawText, "```")
	rawText = strings.TrimSuffix(rawText, "```")
	rawText = strings.TrimSpace(rawText)

	var errorCheck struct {
		Error string `json:"error"`
	}
	if json.Unmarshal([]byte(rawText), &errorCheck) == nil && errorCheck.Error == "no_face" {
		sendError(w, "На фото не обнаружено лицо человека", http.StatusBadRequest)
		return
	}

	var analysis LooksMaxAnalyzeResponse
	if err := json.Unmarshal([]byte(rawText), &analysis); err != nil {
		log.Printf("LooksMax JSON Parse Error: %v, raw: %s", err, rawText)
		sendError(w, "Ошибка формата ответа", http.StatusInternalServerError)
		return
	}

	sendJSON(w, APIResponse{Success: true, Data: analysis}, http.StatusOK)
}

// ==================== LOOKSMAX TRANSFORM ====================

func looksMaxTransformHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Только POST запросы", http.StatusMethodNotAllowed)
		return
	}

	var req LooksMaxTransformRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Неверный формат данных", http.StatusBadRequest)
		return
	}

	if req.ImageBase64 == "" {
		sendError(w, "Нет фото", http.StatusBadRequest)
		return
	}

	imgBytes, err := base64.StdEncoding.DecodeString(req.ImageBase64)
	if err != nil {
		sendError(w, "Ошибка декодирования фото", http.StatusBadRequest)
		return
	}

	tipsText := strings.Join(req.Tips, ". ")
	prompt := fmt.Sprintf(`Улучши внешность человека на фото применив эти изменения: %s. 
Сохрани лицо и черты узнаваемыми. Сделай кожу чище, осанку лучше, общий вид ухоженнее. Реалистично, как профессиональный фотошоп.`, tipsText)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	fw, _ := mw.CreateFormFile("image", "photo.jpg")
	fw.Write(imgBytes)

	mw.WriteField("model", "gpt-image-1")
	mw.WriteField("prompt", prompt)
	mw.WriteField("size", "1024x1024")
	mw.WriteField("response_format", "b64_json")

	mw.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/images/edits", &buf)
	httpReq.Header.Set("Authorization", "Bearer "+openaiKey)
	httpReq.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		sendError(w, "Ошибка запроса к OpenAI: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var imgResp struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &imgResp); err != nil || len(imgResp.Data) == 0 {
		log.Printf("OpenAI Image Error: %s", string(body))
		sendError(w, "Ошибка генерации изображения", http.StatusInternalServerError)
		return
	}

	sendJSON(w, APIResponse{Success: true, Data: LooksMaxTransformResponse{
		ImageBase64: imgResp.Data[0].B64JSON,
	}}, http.StatusOK)
}

// ==================== MAIN ====================

func main() {
	apiKey = os.Getenv("OPENROUTER_KEY")
	if apiKey == "" {
		log.Fatal("❌ OPENROUTER_KEY не установлен")
	}

	openaiKey = os.Getenv("OPENAI_KEY")
	if openaiKey == "" {
		log.Println("⚠️  OPENAI_KEY не установлен — looksmax-transform не будет работать")
	}

	httpClient = &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	rateLimiter = NewRateLimiter(10, time.Minute)

	http.HandleFunc("/health", recoverMiddleware(corsMiddleware(healthHandler)))
	http.HandleFunc("/generate", recoverMiddleware(corsMiddleware(authMiddleware(generateHandler))))
	http.HandleFunc("/tips", recoverMiddleware(corsMiddleware(authMiddleware(tipsHandler))))
	http.HandleFunc("/recovery", recoverMiddleware(corsMiddleware(authMiddleware(recoveryHandler))))
	http.HandleFunc("/looksmax-tip", recoverMiddleware(corsMiddleware(looksMaxTipHandler)))
	http.HandleFunc("/looksmax-analyze", recoverMiddleware(corsMiddleware(looksMaxAnalyzeHandler)))
	http.HandleFunc("/chat", recoverMiddleware(corsMiddleware(authMiddleware(chatHandler))))
	http.HandleFunc("/looksmax-transform", recoverMiddleware(corsMiddleware(looksMaxTransformHandler)))
port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("✅ Сервер запущен на порту %s", port)
	log.Printf("📍 Endpoints: /health, /generate, /tips, /recovery, /looksmax-tip, /looksmax-analyze, /looksmax-transform")
	log.Printf("🔒 Rate Limit: 10 запросов/мин на IP")

	if err := http.ListenAndServe("0.0.0.0:"+port, nil); err != nil {
		log.Fatal("❌ Ошибка запуска сервера: ", err)
	}
}