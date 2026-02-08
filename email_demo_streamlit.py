import streamlit as st
import openai
import streamlit.components.v1 as components
import requests
import numpy as np
from numpy.linalg import norm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. 文本切块 Chunking
def chunk_text(text, chunk_size=300):
    """
    将知识库文本切成多个 chunk，避免一次性塞进 prompt
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# 2. RAG 检索 Retrieval（TF-IDF）
def retrieve_top_chunks_tfidf(chunks, query, top_k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# RAG 检索 Retrieval（embedding）
def get_qwen_embedding(text, api_key):
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-v2"
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Qwen Embedding API error: {response.status_code} - {response.text}")
    result = response.json()
    return np.array(result['output']['embeddings'][0]['embedding'])

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def retrieve_top_chunks_embedding(chunks, query, api_key, top_k=3):
    query_vec = get_qwen_embedding(query, api_key)
    chunk_vectors = [get_qwen_embedding(chunk, api_key) for chunk in chunks]
    similarities = [cosine_sim(query_vec, cv) for cv in chunk_vectors]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# 3. Streamlit 页面配置
st.set_page_config(page_title="腾讯游戏 CRM 智能生成系统", layout="wide")
st.title("🎮 腾讯游戏 CRM 智能邮件生成系统 (RAG版)")
st.markdown("---")

# 4. 侧边栏：API Key + 知识库上传
with st.sidebar:
    st.header("⚙️ 系统配置")

    api_key = ""
    try:
        if "api_key" in st.secrets:
            api_key = st.secrets["api_key"]
            st.success("✅ 已从云端安全加载 API 密钥")
        else:
            api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    except:
        api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    
    dashscope_api_key = ""
    try:
        if "dashscope_api_key" in st.secrets:
            dashscope_api_key = st.secrets["dashscope_api_key"]
            st.success("✅ 已加载 DashScope (Qwen) API 密钥")
        else:
            dashscope_api_key = st.text_input(
                "DashScope API Key（可选，用于语义检索；留空则使用关键词匹配）",
                type="password"
            )
    except:
        dashscope_api_key = st.text_input(
            "DashScope API Key（可选，用于语义检索；留空则使用关键词匹配）",
            type="password"
        )

    st.markdown("---")
    st.header("📚 上传游戏知识库 (RAG)")

    uploaded_file = st.file_uploader("上传游戏 Wiki 或版本指南 (.txt)", type=("txt"))

    kb_content = ""
    if uploaded_file:
        kb_content = uploaded_file.read().decode("utf-8")
        st.success("✅ 知识库已加载")

        st.info(f"知识库长度：{len(kb_content)} 字符")

# 移动端提示
st.markdown("""
<style>
.mobile-upload-tip {
    text-align: center;
    font-size: 13px;
    color: #666;
    margin: 8px 0;
    display: none;
}
@media (max-width: 768px) {
    .mobile-upload-tip {
        display: block;
    }
}
</style>
<div class="mobile-upload-tip">
📱 移动端用户请点击左上角「☰」，打开侧边栏上传 RAG 知识库
</div>
""", unsafe_allow_html=True)

# 5. 主界面：输入 + 输出布局
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 活动简报输入")

    preset_options = {
        "自定义输入": "",
        "🔥 王者荣耀：S34 赛季回归活动": (
            "项目：《王者荣耀》S34 赛季回归活动。\n"
            "目标：针对 30 天未活跃老玩家进行唤醒。\n"
            "权益：登录领‘传说皮肤体验券’。\n"
            "风格：国风暗金主题，深色背景。"
        ),
        "🎁 腾讯新游：赛博春季预热": (
            "项目：新游《星际战魂》预约。\n"
            "卖点：限定传说皮肤 8 折。\n"
            "风格：赛博朋克深黑主题，霓虹紫高亮配色。"
        )
    }

    selected_preset = st.selectbox(
        "💡 快速加载模板：",
        list(preset_options.keys())
    )

    campaign_brief = st.text_area(
        "请在此描述活动内容：",
        value=preset_options[selected_preset],
        height=250
    )

    generate_btn = st.button("🚀 开始 AI 自动生成", use_container_width=True)

with col2:
    st.subheader("📤 AI 邮件预览 (RAG Output)")

    if generate_btn:

        if not api_key:
            st.error("❌ 请先配置 API Key！")

        else:
            try:
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )

                with st.spinner("✍️ 正在生成邮件..."):

                    if kb_content:

                        st.success("📚 检测到知识库，启用 RAG 检索增强模式")

                        # Step 1: Chunking
                        chunks = chunk_text(kb_content)

                        # Step 2: Top-K 检索
                        if dashscope_api_key and dashscope_api_key.strip():
                            st.info("🧠 使用 Qwen Embedding 进行语义检索")
                            try:
                                top_chunks = retrieve_top_chunks_embedding(
                                    chunks, campaign_brief, dashscope_api_key, top_k=3
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Embedding 检索失败，回退到 TF-IDF 关键词匹配：{str(e)}")
                                top_chunks = retrieve_top_chunks_tfidf(chunks, campaign_brief, top_k=3)
                        else:
                            st.info("🔍 未提供 Embedding API Key，使用 TF-IDF 关键词匹配进行检索")
                            top_chunks = retrieve_top_chunks_tfidf(chunks, campaign_brief, top_k=3)

                        retrieved_context = "\n\n".join(top_chunks)

                        st.markdown("### 🔍 检索到的知识片段 (Top-3)")
                        st.code(retrieved_context)

                        prompt = f"""
                            你是一名资深腾讯游戏 CRM 邮件运营专家。

                            请根据【活动简报】并严格参考【检索知识库片段】，生成生产级 HTML 邮件。

                            【活动简报】
                            {campaign_brief}

                            【检索知识库片段】
                            {retrieved_context}

                            要求：
                            - 仅输出 HTML，不要解释
                            - 包含：标题、副标题、正文、CTA按钮、页脚
                            - 使用简洁内联 CSS
                            - CTA 按钮必须是 <a> 标签
                            - 邮件术语必须与知识库一致
                            - 风格要符合游戏调性
                            """

                    else:

                        st.warning("⚠️ 未上传知识库，使用普通 AI 生成模式（试用）")

                        prompt = f"""
                            你是一名资深腾讯游戏 CRM 邮件运营专家。

                            请根据【活动简报】直接生成一封高质量 HTML 游戏营销邮件。

                            【活动简报】
                            {campaign_brief}

                            要求：
                            - 仅输出 HTML，不要解释
                            - 包含：标题、副标题、正文、CTA按钮、页脚
                            - 使用简洁内联 CSS
                            - CTA 按钮必须是 <a> 标签
                            - 风格要符合游戏调性
                            """

                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )

                    html_content = response.choices[0].message.content

                    # Step 4: 预览 + 下载
                    components.html(html_content, height=600, scrolling=True)

                    st.download_button(
                        "💾 下载 HTML 文件",
                        data=html_content,
                        file_name="game_crm_email.html"
                    )

            except Exception as e:
                st.error(f"生成失败：{str(e)}")

