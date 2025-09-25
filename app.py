import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


@st.cache_resource
def load_llm(performance_mode=True, model_choice="llama2"):
    """Load and cache the LLM to avoid reloading on every function call
    
    Args:
        performance_mode: If True, use faster settings with slightly lower quality
        model_choice: Which model to use ("llama2", "mistral", "phi3", "gemma", "tinyllama")
    """
    models = {
        "llama2": {
            "path": './models/llama-2-7b-chat.ggmlv3.q8_0.bin',
            "type": 'llama'
        },
        "mistral": {
            "path": './models/mistral-7b-instruct-v0.2.Q4_K_M.gguf',  
            "type": 'mistral'
        },
        "phi3": {
            "path": './models/phi-3-mini-4k-instruct.Q4_K_M.gguf',  
            "type": 'phi'
        },
        "gemma": {
            "path": './models/gemma-2b-it.Q4_K_M.gguf',  
            "type": 'gemma'
        },
        "tinyllama": {
            "path": './models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', 
            "type": 'llama'
        }
    }
    
    
    if model_choice not in models:
        model_choice = "llama2"
    
    # Select configuration based on performance mode
    if performance_mode:
        return CTransformers(
            model=models[model_choice]["path"],
            model_type=models[model_choice]["type"],
            config={
                'max_new_tokens': 768,    
                'temperature': 0.2,        
                'context_length': 2048, 
                'top_p': 0.85,           
                'threads': 4,          
                'batch_size': 8      
            }
        )
    else:
        # Optimized for quality
        return CTransformers(
            model=models[model_choice]["path"],
            model_type=models[model_choice]["type"],
            config={
                'max_new_tokens': 1536,    
                'temperature': 0.03,    
                'context_length': 4096,
                'top_p': 0.95,            
                'threads': 4,          
                'batch_size': 1           
            }
        )

def getLLamaresponse(blog_topic, target_audience="Researchers", tone="Professional", word_count="500", 
                  content_type="Blog Post", keywords=None, seo_focus=True, performance_mode=True, 
                  include_sources=False, model_choice="llama2"):
    
    llm = load_llm(performance_mode, model_choice)

    keyword_instruction = ""
    if keywords and len(keywords) > 0:
        keyword_instruction = f"Focus on these keywords: {', '.join(keywords)}. "

    template="""
    <s>[INST] <<SYS>>
    You are BlogGPT, an expert blog writer. Create a well-structured {content_type} for {target_audience} audience on "{blog_topic}" in {tone} tone, ~{word_count} words.
    
    Quick audience guide:
    - Researchers: Academic, methodical, evidence-based
    - Data Scientists: Technical, practical, code-oriented
    - Business Professionals: ROI-focused, strategic, professional
    - General Public: Clear, relatable, minimal jargon
    <</SYS>>
    
    Write a {content_type} with:
    1. Title with keyword
    2. Introduction
    3. Main content with H2/H3 headings
    4. Conclusion with call to action
    {keyword_instruction}{seo_instructions}
    {source_instructions}
    [/INST]
    
    # """
    
    seo_instructions = """5. Include SEO keywords and proper headings.""" if seo_focus else ""
    source_instructions = """6. Add 'References' section.""" if include_sources else ""
    
    prompt = PromptTemplate(input_variables=["content_type", "target_audience", "blog_topic", "tone", "word_count", 
                                           "keyword_instruction", "seo_instructions", "source_instructions"],
                           template=template)
    
    response=llm(prompt.format(
        content_type=content_type,
        target_audience=target_audience,
        blog_topic=blog_topic,
        tone=tone,
        word_count=word_count,
        keyword_instruction=keyword_instruction,
        seo_instructions=seo_instructions,
        source_instructions=source_instructions
    ))
    print(response)
    return response

# Streamlit app configuration
st.set_page_config(page_title="Blog Generator",
                    page_icon='📝',
                    layout='centered',
                    initial_sidebar_state='expanded')

with st.sidebar:
    st.title("About")
    st.markdown("""
    # AI Blog Generator 📝
    
    This application uses LLama 2 to generate professional blog posts tailored to your needs.
    
    ### Features:
    - Audience-specific content
    - SEO-optimized structure
    - Customizable word count
    - Tone and style options
    
    ### How to use:
    1. Enter your blog topic
    2. Set the desired word count
    3. Select your target audience
    4. Choose the tone of writing
    5. Click "Generate Blog" and wait for the magic!
    """)
    
    st.info("💡 **Tip**: Be specific with your topic to get better results!", icon="💡")

st.title("AI Blog Generator 📝")
st.subheader("Create professional blogs in seconds")

blog_topic = st.text_area("What would you like to write about?", 
                         placeholder="Enter your blog topic here...",
                         help="Be specific about your topic for better results")

st.markdown("### Blog Settings")

col1, col2 = st.columns([1, 1])

with col1:
    word_count = st.text_input('Word Count', 
                            placeholder="e.g., 500",
                            help="Approximate number of words for your blog")
    
    target_audience = st.selectbox('Target Audience',
                            ('Researchers', 'Data Scientists', 'Business Professionals', 'General Public'),
                            index=0,
                            help="Select your target audience for appropriate tone and detail")

with col2:
    tone = st.selectbox('Writing Tone',
                       ('Professional', 'Conversational', 'Educational', 'Persuasive', 'Inspirational'),
                       index=0,
                       help="Select the tone for your blog")
    
    content_type = st.selectbox('Content Type',
                              ('Blog Post', 'Article', 'Tutorial', 'Review', 'Opinion Piece'),
                              index=0,
                              help="Select the type of content you want to create")
    
# Advanced options expander
with st.expander("Advanced Options"):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        seo_focus = st.checkbox("Optimize for SEO", value=True, 
                              help="Include SEO best practices in blog structure")
        performance_mode = st.checkbox("Speed Mode", value=True,
                               help="Generate blogs faster with slightly lower detail")
        
    with col_adv2:
        include_sources = st.checkbox("Include references", value=False,
                                   help="Add suggested references or sources")
    
    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose a model (requires download)",
        options=["llama2", "mistral", "phi3", "gemma", "tinyllama"],
        index=0,
        help="More efficient models may require downloading. Currently using the existing Llama 2 model.",
        horizontal=True
    )
    
    if model_choice != "llama2":
        st.info(f"💡 **{model_choice}** model selected! This requires downloading the model file first. Run the `download_models.py` script to get this model.", icon="💡")

st.markdown("### Keywords (Optional)")
keywords = st.text_input("Enter keywords separated by commas", 
                       placeholder="e.g., AI, machine learning, data science",
                       help="Keywords to include in the blog")

keywords_list = [k.strip() for k in keywords.split(',')] if keywords else []

submit = st.button("✨ Generate Blog", type="primary", use_container_width=True)

if submit:
    if not blog_topic.strip():
        st.error("⚠️ Please enter a blog topic")
    elif not word_count.strip():
        st.error("⚠️ Please specify the number of words")
    else:
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        progress_placeholder.text("⚙️ Loading model and preparing prompt...")
        progress_bar.progress(10)
        
        progress_placeholder.text("🧠 Processing content generation...")
        progress_bar.progress(30)
        
        try:
            model_exists = True
            if model_choice != "llama2":
                model_path = f"models/{model_choice}.gguf"
                if not os.path.exists(model_path):
                    progress_placeholder.text(f"⚠️ Model {model_choice} not found!")
                    progress_bar.progress(100)
                    st.error(f"The {model_choice} model file was not found. Please run download_models.py to download it first.")
                    model_exists = False
            
            if model_exists:
                response = getLLamaresponse(
                    blog_topic=blog_topic,
                    target_audience=target_audience,
                    tone=tone,
                    word_count=word_count,
                    content_type=content_type,
                    keywords=keywords_list,
                    seo_focus=seo_focus,
                    performance_mode=performance_mode,
                    include_sources=include_sources,
                    model_choice=model_choice
                )
                
                progress_placeholder.text("✅ Blog successfully generated!")
                progress_bar.progress(100)
                
                tab1, tab2 = st.tabs(["📝 Formatted Blog", "🔍 Raw Markdown"])
                
                with tab1:
                    st.markdown(response)
                
                with tab2:
                    st.code(response, language="markdown")
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="📥 Download as TXT",
                        data=response,
                        file_name=f"{blog_topic.replace(' ', '_')}_blog.txt",
                        mime="text/plain"
                    )
                with col_dl2:
                    st.download_button(
                        label="📥 Download as MD",
                        data=response,
                        file_name=f"{blog_topic.replace(' ', '_')}_blog.md",
                        mime="text/markdown"
                    )
                    
                st.divider()
                st.subheader("📊 How was the generated blog?")
                feedback = st.slider("Rate the quality", 1, 5, 3, help="This helps us improve the system")
                
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback! We'll use it to improve the system.")
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Try adjusting your inputs or try again later.")