# AstraTraffic 项目目录结构

基于 LangChain 架构的智慧交通 RAG 系统

> 个人出行管家：从知识问答到出行规划，用 AI 重新定义个人交通出行体验

---

## 📋 项目进展概览

### ✅ 已实现功能

- **🧠 智能决策**：基于 LLM 的智能工具选择，自动路由到知识库或实时工具
- **📚 知识检索增强**：混合检索系统（FAISS + BM25），多知识库支持，引用溯源
- **📡 实时数据**：实时路况查询，智能信息提取，可扩展的实时工具框架
- **🗺️ 实时地图**：交互式地图可视化，地图标记与控制，前后端数据联动
- **💬 多轮对话**：对话历史管理，上下文理解，连续对话支持

### 🚀 下一步规划

- **🗺️ 路径规划**：多路径方案生成，实时路况优化，多交通方式支持，个性化推荐
- **📊 出行预测**：出行时间预测，拥堵预测，天气影响分析，主动建议生成

---

## 📁 根目录文件

```
├── requirements.txt          # Python 依赖包列表
├── README.md                 # 项目说明文档
├── project_structure.md      # 本文件：项目结构说明
```

---

## 🎯 core/ - 核心编排

核心 RAG 流程编排，使用 LangChain LCEL 构建

**业务层归属**：编排层（协调五层协作）

```
core/
├── __init__.py              # 包初始化
└── rag_chain.py             # RAG 核心链编排
                            # - create_rag_chain(): 单轮问答链
                            # - create_rag_chain_with_history(): 多轮对话链
                            # - rag_answer_langchain(): 单轮问答接口
                            # - rag_answer_with_history(): 多轮对话接口
                            # [业务层：编排层 - 协调感知、理解、决策、执行各层协作]
```

---

## 🔌 adapters/ - LangChain 适配器

将底层实现适配为 LangChain 接口

```
adapters/
├── __init__.py              # 包初始化
├── retriever.py            # 检索适配器
│                           # - CustomRetriever: 基础检索适配器
│                           # - ToolAwareRetriever: 工具感知检索适配器
│                           # [业务层：L4 执行层 - 知识检索执行器]
│
├── llm.py                  # LLM 适配器
│                           # - CustomLLM: 自定义 LLM 包装器
│                           # - get_langchain_llm(): 获取 LangChain LLM 实例
│                           # - get_llm_client(): 获取底层 LLM 客户端
│
├── prompt.py               # Prompt 适配器
│                           # - create_rag_prompt(): 创建 RAG Prompt
│                           # - create_rag_prompt_with_history(): 创建带历史的 RAG Prompt
│                           # - format_context(): 格式化上下文
│
└── query_rewriter.py        # 查询改写适配器
                            # - rewrite_query_with_history(): 基于历史改写查询
                            # [业务层：L2 理解层 - 语义理解与查询改写]
```

---

## 🔧 services/ - 业务逻辑

实现 Agent 架构的业务逻辑（不依赖 LangChain 框架）

```
services/
├── __init__.py              # 包初始化
│
├── tool_selector.py        # 工具选择业务逻辑
│                           # - ToolSelector: 工具选择器类
│                           # - select_tool(): 基于 LLM 选择工具
│                           # - 智能路由：根据用户查询自动选择知识库或实时工具
│                           # [业务层：L3 决策层 - 工具选择决策（轻量级）]
│
├── quality.py              # 质量检查业务逻辑
│                           # - check_hits_quality(): 判断检索结果质量
│                           # - 基于置信度阈值筛选检索结果
│                           # [业务层：L3 决策层 - 质量评估]
│
├── fallback.py             # 回退业务逻辑
│                           # - get_no_document_response(): 无文档时的友好答复
│                           # - should_fallback(): 判断是否回退
│                           # - check_has_realtime_info(): 检查是否有实时信息
│                           # - extractive_fallback(): 抽取式兜底（LLM 失败时）
│                           # [业务层：L3 决策层 - 回退策略]
│
├── realtime.py             # 实时工具业务逻辑（已合并）
│                           # - RealtimeToolExecutor: 实时工具执行器
│                           # - execute_realtime_tool(): 执行实时工具
│                           # - get_traffic_client(): 获取实时路况客户端
│                           # - get_road_traffic(): 查询道路实时路况
│                           # - get_location_map(): 查询地图位置信息
│                           # - format_traffic_info(): 格式化路况信息
│                           # - format_map_info_to_dict(): 格式化地图信息
│                           # - extract_road_name_from_query(): 从查询中提取道路名称
│                           # - extract_location_from_query(): 从查询中提取位置信息
│                           # [业务层：L1 感知层（被动感知）+ L4 执行层]
│
└── input_handler.py        # 输入处理业务逻辑
                            # - get_history_manager(): 获取历史管理器
                            # - 提供统一的输入处理接口
```

---

## 🏗️ modules/ - 底层实现

核心功能模块，不依赖特定框架，可独立使用

### modules/config/ - 配置管理

```
modules/config/
├── __init__.py             # 包初始化
├── settings.py             # 全局配置中心
│                           # - LLM 模型配置（OpenAI、Ollama、自定义 HTTP）
│                           # - API 密钥配置
│                           # - 检索参数配置（FAISS、BM25、重排序）
│                           # - 实时路况API配置
│
└── tool_config.py          # 工具配置管理
                            # - ToolID: 工具ID类型定义（Literal类型）
                            # - ToolConfig: 工具配置数据类
                            # - _TOOL_CONFIGS: 所有工具配置
                            # - get_tool_config(): 获取工具配置
                            # - tool_to_kb_name(): 工具ID到知识库名称映射
                            # - tool_to_intent_label(): 工具ID到意图标签映射
```

### modules/rag_types/ - 类型定义

```
modules/rag_types/
├── __init__.py             # 包初始化
└── rag_types.py             # 通用数据类型定义
                            # - IntentTop, IntentResult: 意图识别结果
                            # - Hit: 检索结果
                            # - Metrics: 检索指标
                            # - AnswerPack: 答案包装
                            # - ToolSelection: 工具选择结果
                            # - RealtimeToolResult: 实时工具执行结果
```

### modules/generator/ - 生成器模块

**业务层归属**：L2 理解层

```
modules/generator/
├── prompt.py               # Prompt 模板（底层实现，不依赖 LangChain）
│                           # - RAG_PROMPT_TEMPLATE: RAG 提示词模板
│                           # - RAG_PROMPT_WITH_HISTORY_TEMPLATE: 带历史的 RAG 提示词模板
│                           # - TOOL_SELECTION_PROMPT_TEMPLATE: 工具选择提示词模板
│                           # - ROAD_NAME_EXTRACTION_PROMPT_TEMPLATE: 道路名称提取提示词模板
│                           # - format_hits_to_context(): 格式化检索结果为上下文
│                           # - format_chat_history(): 格式化对话历史
│                           # - build_tool_selection_prompt(): 构建工具选择提示词
│                           # - build_road_name_extraction_prompt(): 构建道路名称提取提示词
│                           # [业务层：L2 理解层 - Prompt 生成与语义理解]
│
├── messages.py             # 用户消息模板（输出给用户的内容）
│                           # - get_realtime_traffic_prefix(): 实时路况信息前缀
│                           # - get_no_road_name_message(): 无法提取道路名称的提示
│                           # - get_traffic_api_failed_message(): API调用失败提示
│                           # - get_traffic_success_message(): 查询成功消息
│
└── query_rewriter.py       # 查询改写模块（使用LLM）
                            # - rewrite_query_with_history(): 基于对话历史改写查询
                            # [业务层：L2 理解层 - 语义理解与查询改写]
```

### modules/chat/ - 对话管理

**业务层归属**：L2 理解层

```
modules/chat/
├── __init__.py             # 包初始化
└── history.py              # 对话历史管理模块（内存存储）
                            # - ChatHistoryManager: 对话历史管理器类
                            # - get_history_manager(): 获取全局历史管理器实例
                            # - 线程安全的会话管理
                            # [业务层：L2 理解层 - 上下文记忆管理]
```

### modules/retriever/ - 检索模块

**业务层归属**：L4 执行层

```
modules/retriever/
└── rag_retriever.py         # RAG 检索核心实现
                            # - DenseEncoder: 稠密向量编码（SentenceTransformer）
                            # - KnowledgeSearcher: 混合检索（FAISS + BM25 + 重排序）
                            # - KBIngestor: 知识库入库工具
                            # [业务层：L4 执行层 - 知识检索实现]
                            #
                            # 注意：知识库入库脚本已移至 scripts/ingest_kb.py
```

### modules/realtime/ - 实时数据模块

**业务层归属**：L1 感知层（被动感知）

```
modules/realtime/
├── __init__.py             # 包初始化
├── traffic.py              # 实时路况模块
│                           # - AmapTrafficAPI: 高德地图路况API客户端
│                           # - TrafficInfo: 路况信息数据类
│                           # [业务层：L1 感知层 - 实时数据获取]
└── map.py                  # 实时地图模块
                            # - AmapMapAPI: 高德地图地图API客户端
                            # - MapInfo, MapLocation: 地图信息数据类
                            # - get_map_client(): 获取地图客户端实例
                            # [业务层：L1 感知层 - 地图数据获取]
```

**功能说明**：
- **实时路况查询**：集成高德地图实时路况 API，支持道路拥堵、限行等实时信息
- **地图数据获取**：支持地点搜索、坐标转换、地图信息格式化
- **可扩展设计**：预留接口，支持未来扩展更多实时数据源（天气、公交等）
- **当前模式**：被动感知（按需调用 API）
- **未来增强**：持续监控、事件检测、主动触发

### modules/intent/ - 意图识别模块（已废弃）

**注意**：意图识别模块已移至 `archive/intent_classification/`。

系统现在使用 LLM 提示词进行工具选择，不再需要训练专门的意图分类模型。

如需查看原始训练代码和模型，请参考 `archive/intent_classification/` 目录。

---

## 📚 data/ - 数据目录

### data/knowledge/ - 知识库原始文件

```
data/knowledge/
├── ev/           # 电动汽车知识库
├── handbook/     # 操作手册知识库
├── health/       # 健康知识库
├── iov/          # 车联网知识库
├── law/          # 法律法规知识库
├── parking/      # 停车知识库
├── report/       # 报告知识库
└── transit/      # 交通知识库
```

### data/models/ - 预训练模型打包

```
data/models/
├── bert-base-chinese/        # 中文 BERT 基础模型（HuggingFace）
└── ds_single/                # LoRA 微调后的意图模型（已废弃，保留用于参考）
```

### data/storage/ - 检索索引存储

每个知识库对应一个子目录，包含：

```
data/storage/
├── <kb_name>/                # 知识库名称（如 ev, health, law 等）
│   ├── index.faiss          # FAISS 向量索引文件
│   ├── meta.jsonl           # 文档元数据（文本、来源等）
│   └── bm25_tokens.json     # BM25 分词索引
```

**注意**：这些索引文件由 `scripts/ingest_kb.py` 生成

---

## 🚀 scripts/ - 脚本目录

```
scripts/
├── cli.py                    # 命令行交互工具
│                             # 用法: python scripts/cli.py
│                             # 提供交互式问答界面，显示详细调试信息
│                             # 适合开发调试和服务器环境使用
│
├── ingest_kb.py              # 知识库入库脚本
│                             # 用法: python scripts/ingest_kb.py --root data/knowledge --include law,parking
│                             # 生成 FAISS 索引、BM25 索引和元数据文件
│
└── test_deps.py              # 依赖检查工具
                              # 用法: python scripts/test_deps.py
                              # 检查 Python 依赖包是否正确安装
```

---

## 🌐 api/ - API 服务层

FastAPI 后端服务，提供 RESTful API 接口

```
api/
├── __init__.py              # 包初始化
├── main.py                  # FastAPI 主应用
│                           # - /api/chat: 聊天接口（支持多轮对话）
│                           # - 请求/响应模型定义
│                           # - 错误处理和日志记录
│
└── run_server.py            # 服务器启动脚本
                            # - 使用 Uvicorn 启动 FastAPI 服务
                            # - 默认端口: 8000
                            # - 支持热重载（开发模式）
```

**功能说明**：
- **聊天接口**：接收用户消息，调用 RAG 系统，返回答案和地图数据
- **数据格式**：支持 JSON 格式的请求和响应
- **地图数据**：返回地图相关信息（位置、坐标等）供前端可视化

---

## 🎨 frontend/ - 前端界面

Vue 3 + Vite 构建的现代化 Web 界面

```
frontend/
├── src/
│   ├── api/
│   │   └── chat.js          # API 客户端（Axios）
│   │
│   ├── components/
│   │   ├── ChatPanel.vue    # 聊天面板组件
│   │   └── MapPanel.vue     # 地图面板组件
│   │
│   ├── config/
│   │   └── amap.js          # 高德地图配置
│   │
│   ├── App.vue              # 主应用组件
│   ├── main.js              # 应用入口
│   └── style.css            # 全局样式
│
├── index.html               # HTML 入口
├── package.json             # 前端依赖配置
└── vite.config.js           # Vite 构建配置
```

**功能说明**：
- **聊天界面**：支持发送消息、显示对话历史、加载状态等
- **地图可视化**：基于高德地图 JavaScript API，实时显示地图和标记
- **数据联动**：后端返回地图数据时，前端自动渲染地图
- **响应式设计**：适配不同屏幕尺寸

---

## 📦 archive/ - 归档目录

存放已废弃但保留作为参考的代码和资源。

```
archive/
├── README.md                 # 归档说明文档
├── .gitignore                # 忽略大文件（模型、数据等）
└── intent_classification/    # 意图识别模块（已废弃）
    ├── intent_classify.py    # 意图分类核心实现
    └── model_training/        # 训练相关代码和数据
        ├── train/            # 训练脚本
        ├── data/             # 训练数据
        └── models/           # 训练好的模型
```

**注意**：
- 此目录中的代码不会被系统使用
- 模型文件较大，已通过 `.gitignore` 配置忽略
- 如需使用，请参考 `archive/README.md`

---

## 📋 架构说明

> **提示**：详细的架构映射说明请参考 [ARCHITECTURE.md](docs/ARCHITECTURE_MAPPING.md)

### 架构映射关系

当前系统采用**技术架构 + 业务架构映射**的设计方式：

- **技术架构**：按代码组织方式划分（`core/`、`adapters/`、`services/`、`modules/`）
- **业务架构**：按智能体能力划分（感知层、理解层、决策层、执行层、反思层）

#### 五层业务架构映射

根据新构想文档（`astratraffic-new.md`），系统应具备五大智能层：

| 业务层 | 技术模块映射 | 实现状态 | 说明 |
|--------|-------------|---------|------|
| **L1 感知层** | `services/realtime.py`<br>`modules/realtime/` | ⭐⭐⭐ | 实时数据获取（被动感知） |
| **L2 理解层** | `adapters/query_rewriter.py`<br>`modules/generator/`<br>`modules/chat/` | ⭐⭐⭐⭐ | 意图识别、查询改写、上下文管理 |
| **L3 决策层** | `services/tool_selector.py`<br>`services/quality.py`<br>`services/fallback.py` | ⭐⭐⭐ | 工具选择决策（轻量级） |
| **L4 执行层** | `adapters/retriever.py`<br>`modules/retriever/`<br>`services/realtime.py` | ⭐⭐⭐⭐ | 知识检索、实时工具执行 |
| **L5 反思层** | ❌ 未实现 | ⭐ | 暂无学习机制 |

**当前模式**：被动响应式（用户查询 → 系统响应）

**未来演进**：主动自主式（持续感知 → 主动决策 → 自主行动）

详细说明请参考 [ARCHITECTURE.md](docs/ARCHITECTURE_MAPPING.md)

---

### 分层设计

```
┌─────────────────────────────────────┐
│  scripts/cli.py                     │  应用层
│  (CLI 交互)                          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  core/rag_chain.py                    │  核心编排层
│  (RAG 流程编排)                        │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼───────┐    ┌────────▼────────┐
│ adapters/│    │  services/       │
│ (适配层)  │    │  (业务逻辑)      │
│          │    │                  │
│ LangChain│    │  - 工具选择      │
│ 接口适配  │    │  - 质量检查      │
└───┬───────┘    │  - 回退逻辑      │
    │            │  - 实时工具      │
    └────────────┴──────────────────┘
               │
┌──────────────▼──────────────────────┐
│  modules/                             │  底层实现层
│  - config/    配置管理                │  - 业务逻辑实现
│  - types/     类型定义                │  - 不依赖特定框架
│  - generator/  Prompt & 消息模板       │  - 可独立使用
│  - chat/      对话历史管理            │
│  - retriever/ FAISS + BM25 检索      │
│  - realtime/  实时API封装             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  data/                               │  数据层
│  - knowledge/                        │  - 原始知识库
│  - models/                           │  - 预训练模型
│  - storage/                          │  - 检索索引
└─────────────────────────────────────┘
```

### 数据流

```
用户输入 
  ↓
工具选择 (LLM 提示词)
  ↓
路由到知识库或实时工具 (根据工具选择)
  ↓
检索 (FAISS + BM25 + Reranker) 或 实时API调用
  ↓
质量检查 (置信度阈值)
  ↓
格式化上下文
  ↓
LLM 生成答案
  ↓
后处理 (添加引用)
  ↓
返回答案
```

---

## 🔑 关键配置文件

- **modules/config/settings.py**: 全局配置（模型路径、API 密钥、检索参数）
- **modules/config/tool_config.py**: 工具配置（工具ID、描述、知识库映射）
- **requirements.txt**: Python 依赖包
- **.env**: 环境变量（通常不提交到 Git，需要单独配置）

---

## 📝 使用说明

### 1. 初始化环境

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env  # 如果有示例文件
# 编辑 .env 填入 API keys 等配置
```

### 2. 准备知识库

```bash
# 入库知识库（示例）
python scripts/ingest_kb.py --root data/knowledge --include law,parking
```

### 3. 运行应用

```bash
# 命令行交互
python scripts/cli.py
```

---

## 🗂️ 文件命名约定

- **适配器**: `adapters/*.py` - LangChain 接口适配
- **业务逻辑**: `services/*.py` - 业务逻辑实现
- **核心编排**: `core/*.py` - RAG 流程编排
- **底层实现**: `modules/*/*.py` - 框架无关的核心实现
- **配置**: `modules/config/*.py` - 配置管理
- **类型**: `modules/rag_types/*.py` - 类型定义

---

## 📌 注意事项

1. **模型文件**: `data/models/` 和 `archive/intent_classification/model_training/models/` 中的模型文件通常较大，建议使用 Git LFS 或单独下载
2. **索引文件**: `data/storage/` 中的索引文件由入库脚本自动生成，首次使用需要先运行入库
3. **环境变量**: `.env` 文件包含敏感信息，不要提交到版本控制
4. **依赖版本**: 建议严格按照 `requirements.txt` 中的版本安装，避免兼容性问题
5. **向后兼容**: `lc/` 目录已废弃，新代码应使用新的目录结构（`adapters/`、`services/`、`core/`）

---
