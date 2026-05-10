const express = require('express');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '7mb' }));
app.use(express.static(path.join(__dirname)));

// ─── Provider Configuration ────────────────────────────────────────────────

const PROVIDERS = {
  openrouter: {
    label: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    defaultModel: 'baidu/cobuddy:free'
  },
  openai: {
    label: 'OpenAI',
    baseURL: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o-mini'
  },
  deepseek: {
    label: 'DeepSeek',
    baseURL: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat'
  },
  dashscope: {
    label: '通义千问 / DashScope',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    defaultModel: 'qwen-plus'
  },
  zhipu: {
    label: '智谱 GLM',
    baseURL: 'https://open.bigmodel.cn/api/paas/v4',
    defaultModel: 'glm-5.1'
  },
  custom: {
    label: '自定义 OpenAI-compatible',
    baseURL: '',
    defaultModel: ''
  }
};

// ─── Utility Functions ─────────────────────────────────────────────────────

function maskApiKey(key) {
  if (!key || key.length <= 8) return '***';
  return key.slice(0, 4) + '...' + key.slice(-4);
}

function resolveAiConfig(aiConfig) {
  if (!aiConfig || !aiConfig.apiKey) {
    return null;
  }
  const provider = aiConfig.provider || 'openrouter';
  const preset = PROVIDERS[provider];

  let baseUrl;
  if (provider === 'custom') {
    baseUrl = aiConfig.baseUrl || '';
    // 自定义 provider 强制 HTTPS
    if (!baseUrl.toLowerCase().startsWith('https://')) {
      console.error('[Security] Custom provider requires HTTPS');
      return null;
    }
  } else if (aiConfig.baseUrl) {
    baseUrl = aiConfig.baseUrl;
  } else if (preset) {
    baseUrl = preset.baseURL;
  } else {
    baseUrl = '';
  }

  const model = aiConfig.model || (preset ? preset.defaultModel : '') || 'unknown';

  return {
    provider,
    baseUrl,
    model,
    apiKey: aiConfig.apiKey
  };
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function callUnifiedProviderWithRetry(messages, useTools, resolvedConfig, maxRetries = 2) {
    const { provider } = resolvedConfig;
    let lastError = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        const response = await callUnifiedProvider(messages, useTools, resolvedConfig);

        if (response.status !== 429) {
            return response;
        }

        lastError = response;

        if (attempt < maxRetries) {
            const delay = Math.pow(2, attempt) * 3000 + Math.random() * 2000;
            console.log(`[429] Rate limited, retrying in ${Math.round(delay / 1000)}s (attempt ${attempt + 1}/${maxRetries})`);
            await sleep(delay);
        }
    }

    console.error(`[429] All ${maxRetries + 1} attempts failed with 429`);
    return lastError;
}

function normalizeError(err, provider, model) {
  let errorCode = 'provider_error';
  let message = 'AI 服务商返回错误';

  if (err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('network'))) {
    errorCode = 'network_error';
    message = '网络连接失败，请检查 Base URL 和网络';
  }

  const statusCode = err.statusCode || err.status;
  if (statusCode) {
    switch (statusCode) {
      case 401:
        errorCode = 'unauthorized';
        message = 'API Key 无效或未配置';
        break;
      case 402:
        errorCode = 'insufficient_balance';
        message = '账户余额不足，请充值后重试';
        break;
      case 403:
        errorCode = 'forbidden';
        message = '无权限访问该资源，请检查账户状态';
        break;
      case 404:
        errorCode = 'not_found';
        message = 'Base URL 或模型名称可能不正确';
        break;
      case 429:
        errorCode = 'rate_limited';
        message = '请求频率超限（RPM），请稍后重试。有余额不代表无频率限制，DeepSeek 等服务商对每分钟请求数有上限。';
        break;
      default:
        if (statusCode >= 500) {
          errorCode = 'provider_error';
          message = 'AI 服务端异常，请稍后重试';
        }
        break;
    }
  }

  return {
    success: false,
    errorCode,
    message,
    provider,
    model
  };
}

function extractJson(text) {
    if (!text || text.trim() === '') return null;
    let s = text.trim();
    s = s.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '');
    try {
        return JSON.parse(s);
    } catch (_) {}
    const first = s.indexOf('{');
    const last = s.lastIndexOf('}');
    if (first !== -1 && last !== -1 && last > first) {
        try {
            return JSON.parse(s.slice(first, last + 1));
        } catch (_) {}
    }
    return null;
}

const AI_SCHEMA_FIELDS = [
    'ai_summary', 'job_type_review', 'fit_review',
    'personalized_strengths', 'main_gaps', 'priority_actions',
    'resume_polish', 'interview_talking_points', 'warnings',
    'strengths', 'gaps', 'actions', 'interview', 'resume_suggestions'
];

const LOCAL_FIELDS = [
    'jobType', 'matchLevel', 'matchedSkills',
    'partialSkills', 'missingSkills', 'infoStatus'
];

function validateAIResult(result) {
    if (!result || typeof result !== 'object') return false;
    const keys = Object.keys(result);
    const hasAIField = keys.some(k => AI_SCHEMA_FIELDS.includes(k));
    const onlyLocalFields = keys.every(k => LOCAL_FIELDS.includes(k));
    if (onlyLocalFields && !hasAIField) return false;
    return hasAIField;
}

function normalizeAIResult(result) {
    if (!result || typeof result !== 'object') return result;

    const strengths = result.strengths || result.personalized_strengths || [];
    const gaps = result.gaps || result.main_gaps || [];
    const actions = result.actions || result.priority_actions || [];
    const interview = result.interview || result.interview_talking_points || [];
    const resumePolish = result.resume_polish || result.resume_suggestions || [];
    const warnings = result.warnings || [];

    const normalized = {
        ai_summary: result.ai_summary || '',
        job_type_review: result.job_type_review || {},
        fit_review: result.fit_review || {},
        strengths: strengths.map(s => ({
            point: s.point || '',
            evidence: s.evidence || '',
            resume_wording: s.resume_wording || s.how_to_write_in_resume || s.suggestion || ''
        })),
        gaps: gaps.map(g => ({
            gap: g.gap || '',
            fix: g.fix || g.student_friendly_fix || g.suggestion || g.why_it_matters || ''
        })),
        actions: actions.map(a => ({
            priority: a.priority || '',
            action: a.action || a.task || '',
            output: a.output || a.expected_output || a.estimated_time || a.time_cost || a.reason || ''
        })),
        resume_polish: resumePolish.map(rp => ({
            before: rp.before || rp.section || '',
            after: rp.after || rp.example || rp.suggestion || '',
            reason: rp.reason || ''
        })),
        interview: interview.map(itp => ({
            question: itp.question || itp.topic || '',
            answer_hint: itp.answer_hint || itp.how_to_prepare || itp.why || ''
        })),
        warnings: warnings.map(w => {
            if (typeof w === 'string') return { warning: w, fix: '' };
            return { warning: w.warning || w.fix || '', fix: w.fix || '' };
        })
    };

    return normalized;
}

async function callUnifiedProvider(messages, useTools, resolvedConfig) {
    const { provider, baseUrl, model, apiKey } = resolvedConfig;

    const body = {
        model,
        messages,
        temperature: 0.2,
        max_tokens: resolvedConfig.max_tokens || 3500,
        stream: false
    };

    if (useTools) {
        body.tools = [{
            type: 'function',
            function: {
                name: 'analyze_job_fit',
                description: '分析学生与岗位的匹配度',
                parameters: {
                    type: 'object',
                    properties: {
                        ai_summary: { type: 'string', description: '综合分析总结，1-2句话' },
                        job_type_review: {
                            type: 'object',
                            properties: {
                                job_type: { type: 'string', enum: ['embedded','electrical_plc','hardware_pcb','hardware_test','robotics','backend','operation','content_operation','unknown','out_of_scope'] },
                                confidence: { type: 'string', enum: ['高','中','低'] },
                                reason: { type: 'string', description: '判断理由，1句话' }
                            },
                            required: ['job_type','confidence','reason']
                        },
                        fit_review: {
                            type: 'object',
                            properties: {
                                score: { type: 'integer', minimum: 0, maximum: 100 },
                                level: { type: 'string', enum: ['高匹配','部分匹配','低匹配'] },
                                reason: { type: 'string', description: '匹配度理由，1句话' }
                            },
                            required: ['score','level','reason']
                        },
                        personalized_strengths: {
                            type: 'array',
                            maxItems: 3,
                            items: {
                                type: 'object',
                                properties: {
                                    point: { type: 'string' },
                                    evidence: { type: 'string' },
                                    how_to_write_in_resume: { type: 'string' }
                                },
                                required: ['point','evidence','how_to_write_in_resume']
                            }
                        },
                        main_gaps: {
                            type: 'array',
                            maxItems: 3,
                            items: {
                                type: 'object',
                                properties: {
                                    gap: { type: 'string' },
                                    why_it_matters: { type: 'string' },
                                    student_friendly_fix: { type: 'string' }
                                },
                                required: ['gap','why_it_matters','student_friendly_fix']
                            }
                        },
                        priority_actions: {
                            type: 'array',
                            maxItems: 3,
                            items: {
                                type: 'object',
                                properties: {
                                    priority: { type: 'string', enum: ['P1','P2','P3'] },
                                    action: { type: 'string' },
                                    time_cost: { type: 'string' },
                                    expected_output: { type: 'string' }
                                },
                                required: ['priority','action','time_cost','expected_output']
                            }
                        },
                        resume_polish: {
                            type: 'array',
                            maxItems: 2,
                            items: {
                                type: 'object',
                                properties: {
                                    before: { type: 'string' },
                                    after: { type: 'string' },
                                    reason: { type: 'string' }
                                },
                                required: ['before','after','reason']
                            }
                        },
                        interview_talking_points: {
                            type: 'array',
                            maxItems: 2,
                            items: {
                                type: 'object',
                                properties: {
                                    question: { type: 'string' },
                                    answer_hint: { type: 'string' }
                                },
                                required: ['question','answer_hint']
                            }
                        },
                        warnings: {
                            type: 'array',
                            maxItems: 2,
                            items: {
                                type: 'object',
                                properties: {
                                    warning: { type: 'string' },
                                    fix: { type: 'string' }
                                },
                                required: ['warning','fix']
                            }
                        }
                    },
                    required: ['ai_summary','job_type_review','fit_review','personalized_strengths','main_gaps','priority_actions','resume_polish','interview_talking_points','warnings']
                }
            }
        }];
        body.tool_choice = 'auto';
    }

    const endpoint = baseUrl.endsWith('/chat/completions')
        ? baseUrl
        : baseUrl.replace(/\/+$/, '') + '/chat/completions';

    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
    };

    if (provider === 'openrouter') {
        headers['HTTP-Referer'] = 'http://localhost:3000';
        headers['X-Title'] = 'Job Path Helper';
    }

    console.log(`[AI] provider=${provider}, model=${model}, baseURL=${baseUrl}, key=${maskApiKey(apiKey)}`);

    const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(body)
    });

    return response;
}

app.post('/api/ai-analyze', async (req, res) => {
    const { jdText, userBackground, localAnalysis, aiConfig: rawAiConfig } = req.body;

    console.log('AI analyze request received');
    console.log('jdText length:', jdText?.length || 0);
    console.log('userBackground length:', userBackground?.length || 0);
    console.log('localAnalysis keys:', Object.keys(localAnalysis || {}));

    let resolvedConfig = resolveAiConfig(rawAiConfig);

    if (!resolvedConfig) {
        if (!process.env.OPENROUTER_API_KEY) {
            console.log('ERROR: aiConfig not provided and OPENROUTER_API_KEY not configured');
            return res.status(500).json({
                success: false,
                errorCode: 'unauthorized',
                message: '未配置 AI 服务，请先在设置中配置 API Key。',
                provider: 'openrouter',
                model: PROVIDERS.openrouter.defaultModel
            });
        }

        const preset = PROVIDERS.openrouter;
        resolvedConfig = {
            provider: 'openrouter',
            baseUrl: preset.baseURL,
            model: preset.defaultModel,
            apiKey: process.env.OPENROUTER_API_KEY
        };
        console.log(`[AI] Fallback to .env: provider=openrouter, model=${resolvedConfig.model}, key=${maskApiKey(resolvedConfig.apiKey)}`);
    }

    if (!jdText && !userBackground) {
        return res.status(400).json({
            error: "请求体不能为空，请至少提供 JD 或个人背景。"
        });
    }

    const systemPrompt = `你是一个大学生实习求职岗位匹配分析助手。
你会收到学生个人背景、岗位 JD，以及本地规则分析结果（localAnalysis）。
localAnalysis 只是参考材料，禁止原样返回。

输出要求：极简，每个数组最多3项，每个字符串不超过30字。

禁止输出 Markdown。禁止输出代码块。禁止输出解释性文字。禁止输出思考过程。
禁止原样返回 localAnalysis。禁止只返回 jobType、matchLevel、matchedSkills、infoStatus 等本地分析字段。

岗位类型只允许：embedded, electrical_plc, hardware_pcb, hardware_test, robotics, backend, operation, content_operation, unknown, out_of_scope

岗位类型定义：
- operation：数据运营/数据分析，JD必须包含SQL/BI/留存/转化率/漏斗分析/指标体系/数据建模等强数据证据
- content_operation：内容/新媒体/社群/活动运营，JD包含公众号/小红书/短视频/文案/Canva/剪映等，但无SQL/BI/留存/转化率等强数据证据
- out_of_scope：非技术非运营方向（HR/行政/财务/法务/翻译/教育/设计等）

特别规则：
1. 如果 localAnalysis 中已有岗位类型，除非 JD 证据明显冲突，否则不要改。
2. 不要把嵌入式软件岗位分析成 PLC、电工证、电气柜、强电、HMI、变频器接线。
3. 如果学生背景有 Arduino、STM32、PWM、小机械臂、嘉立创 EDA、PCB，不要说"完全没有项目基础"。
4. 如果学生没有实习经历，不要编造实习经历。
5. 如果学生代码基础弱，不要强行推荐后端/Web 开发。
6. 建议必须符合本科生实习求职水平，不要写成高级工程师要求。
7. unknown 岗位兜底时，只能引用 JD 原文中出现过的技能词。
8. 输出要简洁、具体、可执行。
9. 如果 localAnalysis.jobType 为 out_of_scope，说明本地规则无法覆盖该方向，请根据用户背景和 JD 独立分析，不要套用工程技术岗位模板。
10. 如果岗位是非技术方向（如新媒体、HR、行政、市场、教育、财务、法务、翻译、设计等），建议必须围绕该方向的实际技能要求，不要推荐嵌入式、PLC、硬件测试、后端等工程技术方向。`;

    const slimAnalysis = {
        jobType: localAnalysis?.jobType,
        matchLevel: localAnalysis?.matchLevel,
        matchedSkills: localAnalysis?.matchedSkills,
        partialSkills: localAnalysis?.partialSkills,
        missingSkills: localAnalysis?.missingSkills,
        infoStatus: localAnalysis?.infoStatus,
    };

    const isOos = localAnalysis?.jobType === 'out_of_scope' || localAnalysis?.infoStatus === 'ai_recommended';

    const userPrompt = `【个人背景】
${userBackground}

【岗位 JD】
${jdText}

【本地规则分析结果（仅供参考，不要原样返回）】
${JSON.stringify(slimAnalysis, null, 2)}

${isOos ? `【重要提示】当前本地规则无法覆盖该方向。请根据用户背景和 JD 独立分析，不要套用工程技术岗位模板。禁止推荐嵌入式、PLC、硬件测试、后端等方向，除非 JD 原文明确要求。建议只围绕 JD 原文和用户背景展开。如果是新媒体/内容/社群/活动运营方向，job_type请填content_operation，建议围绕选题、文案、内容日历、作品集、数据复盘、用户互动等，不要推荐 SQL、Python pandas、AARRR、漏斗分析等数据运营工具。只有JD明确包含SQL/BI/留存/转化率/漏斗分析/指标体系时，job_type才能填operation。` : ''}
请调用 analyze_job_fit 函数返回分析结果。`;

    const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
    ];

    const provider = resolvedConfig.provider;
    const model = resolvedConfig.model;

    try {
        let lastRaw = null;
        let aiResult = null;

        console.log('Attempt 1: function calling');
        try {
            let response = await callUnifiedProviderWithRetry(messages, true, resolvedConfig);
            console.log('AI status:', response.status);

            if (response.ok) {
                let data = await response.json();
                console.log('finish_reason:', data.choices?.[0]?.finish_reason);

                const toolCalls = data.choices?.[0]?.message?.tool_calls;
                if (toolCalls && toolCalls.length > 0 && toolCalls[0].function?.arguments) {
                    console.log('Function calling response received');
                    const argsStr = toolCalls[0].function.arguments;
                    aiResult = extractJson(argsStr);
                    if (!aiResult) {
                        try { aiResult = JSON.parse(argsStr); } catch (_) {}
                    }
                    if (!aiResult) {
                        console.log('Function calling arguments may be truncated, repairing');
                        aiResult = repairTruncatedJson(argsStr);
                    }
                    if (aiResult && validateAIResult(aiResult)) {
                        console.log('Function calling succeeded, keys:', Object.keys(aiResult));
                        return res.json(normalizeAIResult(aiResult));
                    }
                    console.log('Function calling result invalid, checking content');
                }

                const attempt1Content = data.choices?.[0]?.message?.content;
                const attempt1Finish = data.choices?.[0]?.finish_reason;
                if (attempt1Content && attempt1Content.trim()) {
                    console.log('Model returned content (not tool_calls), parsing');
                    aiResult = extractJson(attempt1Content);
                    if (!aiResult && attempt1Finish === 'length') {
                        console.log('Attempt 1 content truncated, repairing');
                        aiResult = repairTruncatedJson(attempt1Content);
                    }
                    lastRaw = attempt1Content;
                    if (aiResult && validateAIResult(aiResult)) {
                        console.log('Attempt 1 content parsed, keys:', Object.keys(aiResult));
                        return res.json(normalizeAIResult(aiResult));
                    }
                    console.log('Attempt 1 content invalid, proceeding to retry');
                }
            } else {
                const errBody = await response.text();
                console.error(`Attempt 1 API error: status=${response.status}`);
            }
        } catch (attempt1Err) {
            console.error('Attempt 1 exception:', attempt1Err.message);
        }

        console.log('Attempt 2: auto-retry with simplified prompt');
        const simpleSystemPrompt = `你是求职分析助手。只返回JSON，禁止Markdown，禁止解释，禁止思考过程。极简输出：数组最多2项，字符串不超过20字。
岗位类型：embedded,electrical_plc,hardware_pcb,hardware_test,robotics,backend,operation,unknown
禁止原样返回localAnalysis。禁止返回jobType/matchLevel/matchedSkills/infoStatus等本地字段。`;

        const simpleUserPrompt = `背景：${userBackground}
JD：${jdText}
本地分析(仅供参考，禁止原样返回)：${JSON.stringify(slimAnalysis)}

返回JSON：{ai_summary,job_type_review:{job_type,confidence,reason},fit_review:{score,level,reason},strengths:[{point,evidence,resume_wording}],gaps:[{gap,fix}],actions:[{priority,action,output}]}`;

        const simpleMessages = [
            { role: 'system', content: simpleSystemPrompt },
            { role: 'user', content: simpleUserPrompt }
        ];

        const retryResponse = await callUnifiedProviderWithRetry(simpleMessages, false, resolvedConfig);

        if (!retryResponse.ok) {
            console.error(`Retry API error: status=${retryResponse.status}`);
            const errInfo = normalizeError({ statusCode: retryResponse.status }, provider, model);
            return res.status(retryResponse.status).json(errInfo);
        }

        const retryData = await retryResponse.json();
        console.log('Retry finish_reason:', retryData.choices?.[0]?.finish_reason);

        const retryContent = retryData.choices?.[0]?.message?.content;
        const retryFinish = retryData.choices?.[0]?.finish_reason;

        if (!retryContent || retryContent.trim() === '') {
            console.error('Retry returned empty content');
            return res.status(500).json({
                success: false,
                errorCode: 'parse_error',
                message: '模型返回内容为空，请重试',
                provider,
                model
            });
        }

        console.log('Retry content length:', retryContent.length);
        aiResult = extractJson(retryContent);

        if (!aiResult && retryFinish === 'length') {
            console.log('Retry content truncated, repairing');
            aiResult = repairTruncatedJson(retryContent);
        }

        if (!aiResult) {
            console.error('Failed to parse retry result as JSON');
            return res.status(500).json({
                success: false,
                errorCode: 'parse_error',
                message: '模型返回格式异常，请重试',
                provider,
                model
            });
        }

        if (Object.keys(aiResult).length === 0) {
            return res.status(500).json({
                success: false,
                errorCode: 'parse_error',
                message: '模型返回内容为空，请重试',
                provider,
                model
            });
        }

        console.log('Retry parsed keys:', Object.keys(aiResult));

        if (!validateAIResult(aiResult)) {
            console.error('Retry result failed schema validation, keys:', Object.keys(aiResult));
            return res.status(500).json({
                success: false,
                errorCode: 'parse_error',
                message: '模型返回结构不符合 AI 增强分析格式，请重试',
                provider,
                model
            });
        }

        console.log('Final response keys:', Object.keys(aiResult));
        res.json(normalizeAIResult(aiResult));

    } catch (error) {
        console.error('AI analysis error:', error.message);
        const errInfo = normalizeError(error, provider, model);
        res.status(500).json(errInfo);
    }
});

function repairTruncatedJson(text) {
    let s = text.trim();
    s = s.replace(/^```json\s*/i, '').replace(/^```\s*/i, '');
    const first = s.indexOf('{');
    if (first === -1) return null;
    s = s.slice(first);

    let braceDepth = 0;
    let bracketDepth = 0;
    let inString = false;
    let escape = false;

    for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        if (escape) { escape = false; continue; }
        if (ch === '\\' && inString) { escape = true; continue; }
        if (ch === '"' && !escape) { inString = !inString; continue; }
        if (inString) continue;
        if (ch === '{') braceDepth++;
        if (ch === '}') braceDepth--;
        if (ch === '[') bracketDepth++;
        if (ch === ']') bracketDepth--;
    }

    if (inString) s += '"';
    while (bracketDepth > 0) { s += ']'; bracketDepth--; }
    while (braceDepth > 0) { s += '}'; braceDepth--; }

    try {
        const result = JSON.parse(s);
        return result;
    } catch (_) {
        return null;
    }
}

// ─── OCR Endpoint ──────────────────────────────────────────────────────────

const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const OCR_MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const OCR_MIN_WIDTH = 300;
const OCR_MIN_HEIGHT = 150;

function parseImageDimensions(buf, mimeType) {
    try {
        if (mimeType === 'image/png') {
            if (buf.length < 24) return null;
            const width = buf.readUInt32BE(16);
            const height = buf.readUInt32BE(20);
            return { width, height };
        }
        if (mimeType === 'image/jpeg') {
            let offset = 0;
            if (buf[0] !== 0xFF || buf[1] !== 0xD8) return null;
            offset = 2;
            while (offset < buf.length - 1) {
                if (buf[offset] !== 0xFF) return null;
                const marker = buf[offset + 1];
                if (marker === 0xC0 || marker === 0xC1 || marker === 0xC2) {
                    if (offset + 9 > buf.length) return null;
                    const height = buf.readUInt16BE(offset + 5);
                    const width = buf.readUInt16BE(offset + 7);
                    return { width, height };
                }
                if ((marker & 0xF0) === 0xD0 && marker !== 0xD0 && marker !== 0xD1 && marker !== 0xD2 && marker !== 0xD3 && marker !== 0xD4 && marker !== 0xD5 && marker !== 0xD6 && marker !== 0xD7 && marker !== 0xD8 && marker !== 0xD9) {
                    offset += 2;
                } else if (marker === 0xD8 || marker === 0xD9) {
                    offset += 2;
                } else if (marker === 0x00) {
                    return null;
                } else {
                    if (offset + 4 > buf.length) return null;
                    const segLen = buf.readUInt16BE(offset + 2);
                    offset += 2 + segLen;
                }
            }
            return null;
        }
        if (mimeType === 'image/webp') {
            if (buf.length < 30) return null;
            if (buf[0] !== 0x52 || buf[1] !== 0x49 || buf[2] !== 0x46 || buf[3] !== 0x46) return null;
            if (buf[8] !== 0x57 || buf[9] !== 0x45 || buf[10] !== 0x42 || buf[11] !== 0x50) return null;
            const chunkFourCC = buf.toString('ascii', 12, 16);
            if (chunkFourCC === 'VP8 ') {
                if (buf.length < 30) return null;
                const width = buf.readUInt16LE(26) & 0x3FFF;
                const height = buf.readUInt16LE(28) & 0x3FFF;
                return { width, height };
            }
            if (chunkFourCC === 'VP8L') {
                if (buf.length < 25) return null;
                const bits = buf.readUInt32LE(21);
                const width = (bits & 0x3FFF) + 1;
                const height = ((bits >> 14) & 0x3FFF) + 1;
                return { width, height };
            }
            return null;
        }
    } catch (_) {
        return null;
    }
    return null;
}

app.post('/api/ocr-jd', async (req, res) => {
    const { imageBase64, mimeType } = req.body;

    console.log('[OCR] request received, mimeType:', mimeType);

    if (!process.env.OPENROUTER_API_KEY) {
        console.error('[OCR] OPENROUTER_API_KEY not configured');
        return res.status(500).json({ error: '截图识别服务未配置，请手动粘贴 JD 文本', errorCode: 'ocr_unavailable' });
    }

    if (!imageBase64 || typeof imageBase64 !== 'string') {
        return res.status(400).json({ error: '请上传有效的图片', errorCode: 'invalid_image' });
    }

    let resolvedMime = mimeType;
    if (!resolvedMime || !ALLOWED_MIME_TYPES.includes(resolvedMime)) {
        return res.status(400).json({ error: '仅支持 PNG、JPG、WebP 格式图片', errorCode: 'invalid_mime' });
    }

    let cleanBase64 = imageBase64;
    if (cleanBase64.startsWith('data:')) {
        const commaIdx = cleanBase64.indexOf(',');
        if (commaIdx !== -1) {
            cleanBase64 = cleanBase64.slice(commaIdx + 1);
        }
    }

    const estimatedSize = Buffer.byteLength(cleanBase64, 'base64');
    console.log('[OCR] mimeType:', resolvedMime, 'size:', Math.round(estimatedSize / 1024), 'KB');

    if (estimatedSize > OCR_MAX_IMAGE_BYTES) {
        return res.status(400).json({ error: '图片过大，请压缩后再上传，或手动粘贴 JD 文本', errorCode: 'image_too_large' });
    }

    const imageBuf = Buffer.from(cleanBase64, 'base64');
    const dims = parseImageDimensions(imageBuf, resolvedMime);
    if (dims) {
        console.log('[OCR] dimensions:', dims.width, 'x', dims.height);
        if (dims.width < OCR_MIN_WIDTH || dims.height < OCR_MIN_HEIGHT) {
            return res.status(400).json({ error: '图片尺寸过小，请上传更清晰的 JD 截图或手动粘贴。', errorCode: 'image_too_small' });
        }
    } else {
        console.log('[OCR] could not parse image dimensions, proceeding anyway');
    }

    try {
        const ocrResponse = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'http://localhost:3000',
                'X-Title': 'Job Path Helper OCR'
            },
            body: JSON.stringify({
                model: 'baidu/qianfan-ocr-fast:free',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: '请逐字识别图片中的招聘信息文字。只输出图片中真实可见文字，不要总结、不要改写、不要补全。如果是招聘截图，即使格式不完整，也请输出可见文字。如果图片完全没有可读文字或明显不是招聘信息，才输出空字符串。'
                        },
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:${resolvedMime};base64,${cleanBase64}`
                            }
                        }
                    ]
                }],
                temperature: 0,
                stream: false,
                max_tokens: 4000
            })
        });

        console.log('[OCR] OpenRouter status:', ocrResponse.status);

        if (!ocrResponse.ok) {
            let errorBody = '';
            try { errorBody = await ocrResponse.text(); } catch (_) {}
            console.error('[OCR] OpenRouter error, status:', ocrResponse.status, 'body:', errorBody.slice(0, 200));
            if (ocrResponse.status === 429) {
                return res.status(429).json({ error: '免费 OCR 模型额度或频率限制，请稍后重试或手动粘贴', errorCode: 'rate_limited' });
            }
            return res.status(502).json({ error: 'OCR 服务暂时不可用，请稍后重试', errorCode: 'provider_error' });
        }

        const data = await ocrResponse.json();
        let text = data.choices?.[0]?.message?.content || '';

        text = text.trim();
        text = text.replace(/^```[\s\S]*?\n/, '').replace(/\n```\s*$/, '');
        text = text.trim();

        console.log('[OCR] result length:', text.length, 'preview:', text.slice(0, 100));

        if (!text) {
            return res.json({ text: '', errorCode: 'ocr_empty' });
        }

        const JD_KEYWORDS = ['岗位职责', '任职要求', '岗位要求', '招聘', '职位', '职责', '要求', '任职', '工作内容', '任职资格', '岗位描述', '薪资', '福利', '经验', '学历', '实习', '助理', '研发', '开发', '测试', '调试', '焊接', '工程师'];
        const hasJdKeyword = JD_KEYWORDS.some(kw => text.includes(kw));
        if (!hasJdKeyword) {
            console.log('[OCR] no JD keyword found, treating as non-JD content');
            return res.json({ text: '', errorCode: 'ocr_filtered' });
        }

        res.json({ text });

    } catch (error) {
        console.error('[OCR] error:', error.message);
        res.status(500).json({ error: '截图识别失败，请手动粘贴 JD 文本', errorCode: 'ocr_unavailable' });
    }
});

// ─── Routes ────────────────────────────────────────────────────────────────

app.post('/api/ai-test', async (req, res) => {
    const { aiConfig: rawAiConfig } = req.body;

    const resolvedConfig = resolveAiConfig(rawAiConfig);

    if (!resolvedConfig) {
        return res.status(400).json({
            success: false,
            errorCode: 'invalid_config',
            message: '请提供有效的 aiConfig（至少包含 provider 和 apiKey）',
            provider: rawAiConfig?.provider || 'unknown',
            model: rawAiConfig?.model || 'unknown'
        });
    }

    const { provider, baseUrl, model } = resolvedConfig;

    if (provider === 'custom' && !baseUrl) {
        return res.status(400).json({
            success: false,
            errorCode: 'invalid_config',
            message: '自定义 provider 必须提供 baseUrl',
            provider,
            model
        });
    }

    const testMessages = [
        { role: 'user', content: 'Hi' }
    ];

    try {
        const response = await callUnifiedProvider(testMessages, false, {
            ...resolvedConfig,
            max_tokens: 16
        });

        if (response.ok) {
            const data = await response.json();
            const content = data.choices?.[0]?.message?.content;
            if (content) {
                return res.json({
                    success: true,
                    provider,
                    model,
                    message: '连接成功'
                });
            } else {
                return res.json({
                    success: false,
                    errorCode: 'provider_error',
                    message: '模型返回内容为空',
                    provider,
                    model
                });
            }
        } else {
            const errInfo = normalizeError({ statusCode: response.status }, provider, model);
            return res.json(errInfo);
        }
    } catch (error) {
        console.error('AI test error:', error.message);
        const errInfo = normalizeError(error, provider, model);
        return res.json(errInfo);
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Job Path Helper server running at http://localhost:${PORT}`);
    console.log('OPENROUTER_API_KEY loaded:', !!process.env.OPENROUTER_API_KEY);
});
