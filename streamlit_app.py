"""
AlphaGenome Streamlit 应用程序
复制 colabs/quick_start.ipynb 笔记本中的核心功能

该应用程序提供以下功能：
1. DNA序列预测
2. 基因组区间预测  
3. 变异效应分析
4. 变异评分
5. 原位诱变分析 (ISM)
6. 可视化功能
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import List, Dict, Any, Optional

# AlphaGenome imports (会在安装后导入)
try:
    from alphagenome.data import gene_annotation
    from alphagenome.data import genome
    from alphagenome.data import transcript as transcript_utils
    from alphagenome.interpretation import ism
    from alphagenome.models import dna_client
    from alphagenome.models import variant_scorers
    from alphagenome.visualization import plot_components
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="AlphaGenome 分析工具",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    color: #2c5aa0;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f4e79;
    margin: 1rem 0;
}

.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
}

.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #ffc107;
    margin: 1rem 0;
}

.error-box {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #dc3545;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 组织类型映射：用户友好名称 -> UBERON ID
ONTOLOGY_TERM_MAP: Dict[str, str] = {
    "肺 (Lung)": "UBERON:0002048",          # Lung
    "大脑 (Brain)": "UBERON:0000955",        # Brain
    "右肝叶 (Right liver lobe)": "UBERON:0001114",  # Right liver lobe
    "结肠 - 横结肠 (Colon - Transverse)": "UBERON:0001157",  # Colon - Transverse
    "小脑 (Cerebellum)": "UBERON:0002037",  # Cerebellum
    "脑干 (Brainstem)": "UBERON:0002298",  # Brainstem
    "脊髓 (Spinal cord)": "UBERON:0002240",  # Spinal cord
    "眼 (Eye)": "UBERON:0000970",  # Eye
    "内耳 (Inner ear)": "UBERON:0006860",  # Inner ear
    "心脏 (Heart)": "UBERON:0000948",  # Heart
    "气管 (Trachea)": "UBERON:0003126",  # Trachea
    "喉 (Larynx)": "UBERON:0001737",  # Larynx
    "咽 (Pharynx)": "UBERON:0000340",  # Pharynx
    "胃 (Stomach)": "UBERON:0000945",  # Stomach
    "小肠 (Small intestine)": "UBERON:0002108",  # Small intestine
    "十二指肠 (Duodenum)": "UBERON:0002114",  # Duodenum
    "空肠 (Jejunum)": "UBERON:0002115",  # Jejunum
    "回肠 (Ileum)": "UBERON:0002116",  # Ileum
    "大肠 (Large intestine)": "UBERON:0000160",  # Large intestine
    "结肠 (Colon)": "UBERON:0001155",  # Colon
    "直肠 (Rectum)": "UBERON:0001052",  # Rectum
    "肝 (Liver)": "UBERON:0002107",  # Liver
    "胆囊 (Gallbladder)": "UBERON:0002110",  # Gallbladder
    "胰腺 (Pancreas)": "UBERON:0001264",  # Pancreas
    "脾 (Spleen)": "UBERON:0002106",  # Spleen
    "肾 (Kidney)": "UBERON:0002113",  # Kidney
    "输尿管 (Ureter)": "UBERON:0000056",  # Ureter
    "膀胱 (Urinary bladder)": "UBERON:0001255",  # Urinary bladder
    "尿道 (Urethra)": "UBERON:0000057",  # Urethra
    "甲状腺 (Thyroid gland)": "UBERON:0001132",  # Thyroid gland
    "副甲状腺 (Parathyroid gland)": "UBERON:0002260",  # Parathyroid gland
    "肾上腺 (Adrenal gland)": "UBERON:0002369",  # Adrenal gland
    "垂体 (Pituitary gland)": "UBERON:0000007",  # Pituitary gland
    "胸腺 (Thymus)": "UBERON:0001178",  # Thymus
    "松果体 (Pineal gland)": "UBERON:0000986",  # Pineal gland
    "卵巢 (Ovary)": "UBERON:0000992",  # Ovary
    "子宫 (Uterus)": "UBERON:0000995",  # Uterus
    "阴道 (Vagina)": "UBERON:0000996",  # Vagina
    "睾丸 (Testis)": "UBERON:0000473",  # Testis
    "前列腺 (Prostate gland)": "UBERON:0002367",  # Prostate gland
    "精囊 (Seminal vesicle)": "UBERON:0001049",  # Seminal vesicle
    "阴茎 (Penis)": "UBERON:0000464",  # Penis
    "皮肤 (Skin)": "UBERON:0002097",  # Skin
    "骨（器官级） (Bone organ)": "UBERON:0001474",  # Bone organ
    "骨骼肌器官 (Skeletal muscle organ)": "UBERON:0001134",  # Skeletal muscle organ
}

def main():
    """主应用程序函数"""
    
    # 主标题
    st.markdown('<h1 class="main-header">🧬 AlphaGenome 分析工具</h1>', unsafe_allow_html=True)
    
    # 检查AlphaGenome是否可用
    if not ALPHAGENOME_AVAILABLE:
        st.markdown("""
        <div class="error-box">
        <h3>⚠️ AlphaGenome 未安装</h3>
        <p>请先安装 AlphaGenome 包：</p>
        <code>pip install alphagenome</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # 侧边栏 - API密钥配置
    with st.sidebar:
        st.markdown("### 🔑 API 配置")
        api_key = st.text_input(
            "AlphaGenome API Key",
            type="password",
            help="输入您的 AlphaGenome API 密钥"
        )
        
        if api_key:
            try:
                # 初始化DNA模型
                if 'dna_model' not in st.session_state:
                    st.session_state.dna_model = dna_client.create(api_key)
                st.success("✅ API 密钥已验证")
            except Exception as e:
                st.error(f"❌ API 密钥验证失败: {str(e)}")
                st.stop()
        else:
            st.warning("⚠️ 请输入 API 密钥以继续")
            st.stop()
    
    # 主要功能选择
    st.markdown("### 📋 选择分析功能")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧬 DNA序列预测", 
        "🗺️ 基因组区间预测", 
        "🔬 变异效应分析", 
        "📊 变异评分", 
        "🔍 原位诱变分析"
    ])
    
    with tab1:
        dna_sequence_prediction()
    
    with tab2:
        genomic_interval_prediction()
    
    with tab3:
        variant_effect_analysis()
    
    with tab4:
        variant_scoring()
    
    with tab5:
        ism_analysis()

def dna_sequence_prediction():
    """DNA序列预测功能"""
    st.markdown('<h2 class="sub-header">🧬 DNA序列预测</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>输入DNA序列来获取AlphaGenome的预测结果。模型可以预测多种输出类型，包括DNase、CAGE、RNA-seq等。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 输入控制
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # DNA序列输入
        sequence_input = st.text_area(
            "DNA序列",
            value="GATTACA",
            height=100,
            help="输入DNA序列（将自动填充到模型支持的长度）",
            key="dna_sequence_input"
        )
        
        # 序列长度选择
        sequence_length = st.selectbox(
            "序列长度",
            options=[2048, 8192, 32768, 131072, 524288, 1048576],
            index=0,
            help="选择模型输入序列长度",
            key="dna_seq_length"
        )
    
    with col2:
        # 输出类型选择
        output_types = st.multiselect(
            "输出类型",
            options=['ATAC', 'CAGE', 'DNASE', 'RNA_SEQ', 'CHIP_HISTONE', 'CHIP_TF', 'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', 'PROCAP'],
            default=['DNASE'],
            help="选择要预测的输出类型",
            key="dna_seq_output_types"
        )
        
        # 组织类型选择
        ontology_term_labels = st.multiselect(
            "组织类型",
            options=list(ONTOLOGY_TERM_MAP.keys()),
            default=["肺 (Lung)"],
            help="选择要分析的组织类型",
            key="dna_seq_ontology_terms"
        )
        
        # 物种选择
        organism = 'HOMO_SAPIENS'
        # organism = st.selectbox(
        #     "物种",
        #     options=['HOMO_SAPIENS', 'MUS_MUSCULUS'],
        #     index=0,
        #     help="选择预测的物种",
        #     key="dna_seq_organism"
        # )
    
    # 预测按钮
    if st.button("🚀 开始预测", key="dna_predict"):
        if not sequence_input.strip():
            st.error("请输入DNA序列")
            return
        
        if not output_types:
            st.error("请选择至少一个输出类型")
            return
        
        if not ontology_term_labels:
            st.error("请选择至少一个组织类型")
            return
        
        # 将可读名称转换为 UBERON ID
        ontology_terms = [ONTOLOGY_TERM_MAP[label] for label in ontology_term_labels]
        
        try:
            with st.spinner("正在进行预测..."):
                # 处理序列
                padded_sequence = sequence_input.strip().upper().center(sequence_length, 'N')
                
                # 转换输出类型
                requested_outputs = [getattr(dna_client.OutputType, ot) for ot in output_types]
                
                # 转换物种
                organism_obj = getattr(dna_client.Organism, organism)
                
                # 进行预测
                output = st.session_state.dna_model.predict_sequence(
                    sequence=padded_sequence,
                    organism=organism_obj,
                    requested_outputs=requested_outputs,
                    ontology_terms=ontology_terms,
                )
                
                # 显示结果
                display_prediction_results(output, output_types)
                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")

def genomic_interval_prediction():
    """基因组区间预测功能"""
    st.markdown('<h2 class="sub-header">🗺️ 基因组区间预测</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>基于基因组坐标或基因符号进行预测分析。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 输入方式选择
    input_method = st.radio(
        "输入方式",
        options=["基因符号", "基因组坐标"],
        index=0,
        key="interval_input_method"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_method == "基因符号":
            gene_symbol = st.text_input(
                "基因符号",
                value="CYP2B6",
                help="输入HGNC基因符号",
                key="interval_gene_symbol"
            )
        else:
            chromosome = st.text_input("染色体", value="chr22", key="interval_chr")
            start_pos = st.number_input("起始位置", value=36000000, min_value=1, key="interval_start")
            end_pos = st.number_input("结束位置", value=36100000, min_value=1, key="interval_end")
        
        # 序列长度
        sequence_length = st.selectbox(
            "序列长度",
            options=[131072, 524288, 1048576],
            index=2,
            help="选择预测的序列长度",
            key="interval_seq_length"
        )
    
    with col2:
        # 输出类型
        output_types = st.multiselect(
            "输出类型",
            options=['RNA_SEQ'],
            default=['RNA_SEQ'],
            key="interval_output_types"
        )
        
        # 组织类型
        ontology_term_labels = st.multiselect(
            "组织类型",
            options=list(ONTOLOGY_TERM_MAP.keys()),
            default=["肺 (Lung)"],
            key="interval_ontology_terms"
        )
        
        # 物种
        organism = 'HOMO_SAPIENS'
        # organism = st.selectbox(
        #     "物种",
        #     options=['HOMO_SAPIENS', 'MUS_MUSCULUS'],
        #     index=0,
        #     key="interval_organism"
        # )
    
    # 预测按钮
    if st.button("🚀 开始预测", key="interval_predict"):
        try:
            with st.spinner("正在进行预测..."):
                # 创建基因组区间
                if input_method == "基因符号":
                    if not gene_symbol.strip():
                        st.error("请输入基因符号")
                        return
                    
                    # 加载GTF文件
                    @st.cache_data
                    def load_gtf():
                        return pd.read_feather(
                            'https://storage.googleapis.com/alphagenome/reference/gencode/'
                            'hg38/gencode.v46.annotation.gtf.gz.feather'
                        )
                    
                    gtf = load_gtf()
                    interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol.strip())
                else:
                    if not chromosome or start_pos >= end_pos:
                        st.error("请输入有效的基因组坐标")
                        return
                    
                    interval = genome.Interval(chromosome, start_pos, end_pos)
                
                # 调整序列长度
                interval = interval.resize(sequence_length)
                
                # 转换参数
                requested_outputs = [getattr(dna_client.OutputType, ot) for ot in output_types]
                organism_obj = getattr(dna_client.Organism, organism)
                ontology_terms = [ONTOLOGY_TERM_MAP[label] for label in ontology_term_labels]
                
                # 进行预测
                output = st.session_state.dna_model.predict_interval(
                    interval=interval,
                    organism=organism_obj,
                    requested_outputs=requested_outputs,
                    ontology_terms=ontology_terms,
                )
                # 显示基因注释
                if input_method == "基因符号":
                    longest_transcripts = display_gene_annotation(interval, gtf)
                
                # 显示结果
                display_prediction_results(output, output_types)

                fig = plot_components.plot(
                    components=[
                        plot_components.TranscriptAnnotation(longest_transcripts),
                        plot_components.Tracks(output.rna_seq),
                    ],
                    interval=output.rna_seq.interval,
                )
                st.pyplot(fig)
                plt.close(fig)  # 防止内存泄漏

                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")

def variant_effect_analysis():
    """变异效应分析功能"""
    st.markdown('<h2 class="sub-header">🔬 变异效应分析</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>分析遗传变异对基因表达和其他基因组功能的影响。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 变异信息输入
        chromosome = st.text_input("染色体", value="chr22", key="variant_chr")
        position = st.number_input("位置", value=36201698, min_value=1, key="variant_pos")
        reference_bases = st.text_input("参考碱基", value="A", key="variant_ref")
        alternate_bases = st.text_input("变异碱基", value="C", key="variant_alt")
        
        # 序列长度
        sequence_length = st.selectbox(
            "序列长度",
            options=[131072, 524288, 1048576],
            index=2,
            key="variant_seq_length"
        )
    
    with col2:
        # 输出类型
        output_types = st.multiselect(
            "输出类型",
            options=['RNA_SEQ'],
            default=['RNA_SEQ'],
            key="variant_output_types"
        )
        
        # 组织类型
        ontology_term_labels = st.multiselect(
            "组织类型",
            options=list(ONTOLOGY_TERM_MAP.keys()),
            default=["肺 (Lung)"],
            key="variant_ontology_terms"
        )
    
    # 分析按钮
    if st.button("🔬 分析变异效应", key="variant_analysis"):
        if not all([chromosome, position, reference_bases, alternate_bases]):
            st.error("请填写完整的变异信息")
            return

        # 加载GTF文件
        @st.cache_data
        def load_gtf():
            return pd.read_feather(
                'https://storage.googleapis.com/alphagenome/reference/gencode/'
                'hg38/gencode.v46.annotation.gtf.gz.feather'
            )
        gtf = load_gtf()
        
        try:
            with st.spinner("正在分析变异效应..."):
                # 创建变异对象
                variant = genome.Variant(
                    chromosome=chromosome,
                    position=position,
                    reference_bases=reference_bases,
                    alternate_bases=alternate_bases,
                )
                
                # 创建区间
                interval = variant.reference_interval.resize(sequence_length)
                
                # 转换参数
                requested_outputs = [getattr(dna_client.OutputType, ot) for ot in output_types]
                ontology_terms = [ONTOLOGY_TERM_MAP[label] for label in ontology_term_labels]
                
                # 进行变异预测
                variant_output = st.session_state.dna_model.predict_variant(
                    interval=interval,
                    variant=variant,
                    requested_outputs=requested_outputs,
                    ontology_terms=ontology_terms,
                )
                # 显示基因注释
                longest_transcripts = display_gene_annotation(interval, gtf)
                
                # 显示结果
                display_variant_results(variant_output, variant, output_types)

                fig = plot_components.plot(
                    [
                        plot_components.TranscriptAnnotation(longest_transcripts),
                        plot_components.OverlaidTracks(
                            tdata={
                                'REF': variant_output.reference.rna_seq,
                                'ALT': variant_output.alternate.rna_seq,
                            },
                            colors={'REF': 'dimgrey', 'ALT': 'red'},
                        ),
                    ],
                    interval=variant_output.reference.rna_seq.interval.resize(2**15),
                    # Annotate the location of the variant as a vertical line.
                    annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
                )
                st.pyplot(fig)
                plt.close(fig)  # 防止内存泄漏
                
        except Exception as e:
            st.error(f"变异分析过程中出现错误: {str(e)}")

def variant_scoring():
    """变异评分功能"""
    st.markdown('<h2 class="sub-header">📊 变异评分</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>使用推荐的评分器对遗传变异进行量化评分。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 变异信息
        chromosome = st.text_input("染色体", value="chr22", key="score_chr")
        position = st.number_input("位置", value=36201698, min_value=1, key="score_pos")
        reference_bases = st.text_input("参考碱基", value="A", key="score_ref")
        alternate_bases = st.text_input("变异碱基", value="C", key="score_alt")
    
    with col2:
        # 评分器选择
        scorer_type = st.selectbox(
            "评分器类型",
            options=['ATAC', 'CAGE', 'DNASE', 'RNA_SEQ', 'CHIP_HISTONE', 'CHIP_TF', 'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', 'CONTACT_MAPS', 'PROCAP'],
            index=3,
            key="score_type"
        )
        
        # 区间长度
        sequence_length = st.selectbox(
            "序列长度",
            options=[131072, 524288, 1048576],
            index=2,
            key="score_length"
        )
    
    # 评分按钮
    if st.button("📊 计算变异评分", key="variant_scoring"):
        try:
            with st.spinner("正在计算变异评分..."):
                # 创建变异和区间
                variant = genome.Variant(
                    chromosome=chromosome,
                    position=position,
                    reference_bases=reference_bases,
                    alternate_bases=alternate_bases,
                )
                
                interval = variant.reference_interval.resize(sequence_length)
                
                # 选择推荐的评分器
                variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer_type]
                
                # 计算评分
                variant_scores = st.session_state.dna_model.score_variant(
                    interval=interval,
                    variant=variant,
                    variant_scorers=[variant_scorer]
                )
                
                # 显示评分结果
                display_scoring_results(variant_scores, variant, scorer_type)
                
        except Exception as e:
            st.error(f"变异评分过程中出现错误: {str(e)}")

def ism_analysis():
    """原位诱变分析功能"""
    st.markdown('<h2 class="sub-header">🔍 原位诱变分析 (ISM)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>通过系统性突变分析来识别DNA序列中的重要功能区域。</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <p><strong>注意：</strong> ISM分析计算量较大，建议使用较短的序列长度以获得更快的结果。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 基因组区间设置
        chromosome = st.text_input("染色体", value="chr20", key="ism_chr")
        start_pos = st.number_input("起始位置", value=3753000, min_value=1, key="ism_start")
        end_pos = st.number_input("结束位置", value=3753400, min_value=1, key="ism_end")
        
        # 序列长度
        sequence_length = st.selectbox(
            "上下文序列长度",
            options=[2048, 8192],
            index=0,
            help="用于预测的上下文序列长度",
            key="ism_seq_length"
        )
        
        # ISM区间长度
        ism_width = st.slider(
            "ISM分析宽度",
            min_value=64,
            max_value=512,
            value=256,
            step=64,
            help="要进行系统性突变的区域宽度",
            key="ism_width"
        )
    
    with col2:
        # 输出类型
        output_type = st.selectbox(
            "输出类型",
            options=['ATAC', 'CAGE', 'DNASE', 'RNA_SEQ', 'CHIP_HISTONE', 'CHIP_TF', 'SPLICE_SITES', 'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', 'CONTACT_MAPS', 'PROCAP'],
            index=3,  # RNA_SEQ的索引是3
            key="ism_output_type"
        )
        
        # 评分宽度
        scoring_width = st.slider(
            "评分宽度",
            min_value=101,
            max_value=1001,
            value=501,
            step=100,
            help="用于评分的窗口宽度",
            key="ism_scoring_width"
        )
        
        # 聚合类型
        aggregation_type = st.selectbox(
            "聚合类型",
            options=['DIFF_MEAN', 'DIFF_MAX', 'ALT_MEAN'],
            index=0,
            key="ism_aggregation_type"
        )
    
    # 分析按钮
    if st.button("🔍 开始ISM分析", key="ism_analysis"):
        if start_pos >= end_pos:
            st.error("起始位置必须小于结束位置")
            return
        
        try:
            with st.spinner("正在进行ISM分析（这可能需要几分钟）..."):
                # 创建序列区间
                sequence_interval = genome.Interval(chromosome, start_pos, end_pos)
                sequence_interval = sequence_interval.resize(sequence_length)
                
                # 创建ISM区间
                ism_interval = sequence_interval.resize(ism_width)
                
                # 创建变异评分器
                variant_scorer = variant_scorers.CenterMaskScorer(
                    requested_output=getattr(dna_client.OutputType, output_type),
                    width=scoring_width,
                    aggregation_type=getattr(variant_scorers.AggregationType, aggregation_type),
                )
                
                # 进行ISM分析
                variant_scores = st.session_state.dna_model.score_ism_variants(
                    interval=sequence_interval,
                    ism_interval=ism_interval,
                    variant_scorers=[variant_scorer],
                )
                
                # 显示ISM结果
                display_ism_results(variant_scores, ism_interval)
                
        except Exception as e:
            st.error(f"ISM分析过程中出现错误: {str(e)}")

def display_prediction_results(output, output_types):
    """显示预测结果"""
    st.markdown('<h3 class="sub-header">📊 预测结果</h3>', unsafe_allow_html=True)
    
    for output_type in output_types:
        output_type_lower = output_type.lower()
        track_data = getattr(output, output_type_lower)
        
        st.markdown(f"#### {output_type} 预测结果")
        
        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("序列长度", track_data.values.shape[0])
        with col2:
            st.metric("轨道数量", track_data.values.shape[1])
        with col3:
            st.metric("平均值", f"{track_data.values.mean():.4f}")
        
        # 显示元数据
        with st.expander(f"查看 {output_type} 轨道元数据"):
            st.dataframe(track_data.metadata, use_container_width=True)
        
        
        # 提供下载选项
        csv_data = pd.DataFrame(track_data.values)
        csv_string = csv_data.to_csv(index=False)
        st.download_button(
            label=f"下载 {output_type} 预测数据 (CSV)",
            data=csv_string,
            file_name=f"{output_type}_predictions.csv",
            mime="text/csv",
            key=f"download_{output_type}"
        )

def display_variant_results(variant_output, variant, output_types):
    """显示变异效应结果"""
    st.markdown('<h3 class="sub-header">📊 变异效应结果</h3>', unsafe_allow_html=True)
    
    # 显示变异信息
    st.markdown(f"**分析的变异:** {variant}")
    
    for output_type in output_types:
        output_type_lower = output_type.lower()
        ref_data = getattr(variant_output.reference, output_type_lower)
        alt_data = getattr(variant_output.alternate, output_type_lower)
        
        st.markdown(f"#### {output_type} 变异效应")
        
        # 计算差异
        diff_values = alt_data.values - ref_data.values
        
        # 显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均差异", f"{diff_values.mean():.6f}")
        with col2:
            st.metric("最大差异", f"{diff_values.max():.6f}")
        with col3:
            st.metric("最小差异", f"{diff_values.min():.6f}")
        

def display_scoring_results(variant_scores, variant, scorer_type):
    """显示变异评分结果"""
    st.markdown('<h3 class="sub-header">📊 变异评分结果</h3>', unsafe_allow_html=True)
    
    # 显示变异信息
    st.markdown(f"**评分的变异:** {variant}")
    
    variant_scores = variant_scores[0]
    
    # 显示基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("基因数量", variant_scores.X.shape[0])
    with col2:
        st.metric("轨道数量", variant_scores.X.shape[1])
    with col3:
        st.metric("总评分数", variant_scores.X.size)
    
    # 显示基因信息
    if scorer_type == 'RNA_SEQ':
        st.markdown("#### 基因评分信息")
        gene_info = variant_scores.obs.copy()
        st.dataframe(gene_info, use_container_width=True)

    # 显示变异评分数据
    st.markdown("#### 变异评分数据的可视化")
    tidy_scores = variant_scorers.tidy_scores([variant_scores], match_gene_strand=True)
    st.dataframe(tidy_scores, use_container_width=True)
    
    # 提供下载选项
    csv_string = tidy_scores.to_csv(index=False)
    st.download_button(
        label="下载变异评分数据 (CSV)",
        data=csv_string,
        file_name="variant_scores.csv",
        mime="text/csv"
    )

def display_ism_results(variant_scores, ism_interval):
    """显示ISM分析结果"""
    st.markdown('<h3 class="sub-header">📊 原位诱变分析结果</h3>', unsafe_allow_html=True)
    
    st.markdown(f"**分析区间:** {ism_interval}")
    st.markdown(f"**变异总数:** {len(variant_scores)}")
    
    # 提取K562细胞系的评分（如果可用）
    def extract_first_track(adata):
        """提取第一个轨道的评分"""
        values = adata.X[:, 0]  # 使用第一个轨道
        return values.flatten()[0]
    
    try:
        # 创建ISM矩阵
        ism_result = ism.ism_matrix(
            [extract_first_track(x[0]) for x in variant_scores],
            variants=[v[0].uns['variant'] for v in variant_scores],
        )
        
        st.markdown("#### ISM贡献评分矩阵")
        
        # 显示矩阵形状信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("位置数", ism_result.shape[0])
        with col2:
            st.metric("碱基类型", ism_result.shape[1])
        with col3:
            st.metric("最大贡献", f"{ism_result.max():.6f}")
        
        # 绘制序列logo
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # 创建简化的序列logo可视化
        positions = range(ism_result.shape[0])
        base_colors = {'A': 'red', 'T': 'blue', 'G': 'orange', 'C': 'green'}
        bases = ['A', 'T', 'G', 'C']
        
        # 对于每个位置，显示贡献最大的碱基
        max_contributions = []
        max_bases = []
        
        for pos in positions:
            pos_scores = ism_result[pos, :]
            max_idx = np.argmax(np.abs(pos_scores))
            max_contributions.append(pos_scores[max_idx])
            max_bases.append(bases[max_idx])
        
        # 绘制贡献图
        colors = [base_colors[base] for base in max_bases]
        bars = ax.bar(positions, max_contributions, color=colors, alpha=0.7)
        
        # 在每个条形上标注碱基
        for i, (bar, base) in enumerate(zip(bars, max_bases)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   base, ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('位置')
        ax.set_ylabel('ISM贡献评分')
        ax.set_title('原位诱变贡献评分')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=base) 
                          for base, color in base_colors.items()]
        ax.legend(handles=legend_elements, title='碱基类型')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # 显示评分统计
        st.markdown("#### 评分统计信息")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均评分", f"{np.mean(max_contributions):.6f}")
        with col2:
            st.metric("最大正向影响", f"{np.max(max_contributions):.6f}")
        with col3:
            st.metric("最大负向影响", f"{np.min(max_contributions):.6f}")
        with col4:
            st.metric("评分范围", f"{np.max(max_contributions) - np.min(max_contributions):.6f}")
        
        # 找出影响最大的位置
        st.markdown("#### 关键功能位置")
        
        # 找出绝对值最大的几个位置
        abs_contributions = np.abs(max_contributions)
        top_indices = np.argsort(abs_contributions)[-10:][::-1]  # 前10个
        
        top_positions_data = []
        for idx in top_indices:
            top_positions_data.append({
                '位置': idx,
                '碱基': max_bases[idx],
                '贡献评分': max_contributions[idx],
                '绝对贡献': abs_contributions[idx]
            })
        
        top_df = pd.DataFrame(top_positions_data)
        st.dataframe(top_df, use_container_width=True)
        
        # 提供下载选项
        ism_df = pd.DataFrame(ism_result, columns=['A', 'T', 'G', 'C'])
        ism_df['位置'] = range(len(ism_df))
        ism_df['最大贡献碱基'] = max_bases
        ism_df['最大贡献值'] = max_contributions
        
        csv_string = ism_df.to_csv(index=False)
        st.download_button(
            label="下载ISM分析结果 (CSV)",
            data=csv_string,
            file_name="ism_analysis_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"ISM结果处理出现错误: {str(e)}")
        
        # 显示原始评分信息
        st.markdown("#### 原始变异评分")
        scores_summary = []
        for i, variant_score in enumerate(variant_scores[:10]):  # 只显示前10个
            variant_info = variant_score[0].uns['variant']
            score_values = variant_score[0].X.flatten()
            scores_summary.append({
                '变异': str(variant_info),
                '平均评分': score_values.mean(),
                '最大评分': score_values.max(),
                '最小评分': score_values.min()
            })
        
        summary_df = pd.DataFrame(scores_summary)
        st.dataframe(summary_df, use_container_width=True)

def display_gene_annotation(interval, gtf):
    """显示基因注释信息"""
    try:
        # 过滤基因组区间内的基因
        gtf_filtered = gene_annotation.filter_protein_coding(gtf)
        gtf_transcripts = gene_annotation.filter_to_longest_transcript(gtf_filtered)
        transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)
        
        longest_transcripts = transcript_extractor.extract(interval)
        
        # if longest_transcripts:
        #     st.markdown("#### 基因注释信息")
        #     st.markdown(f"在区间 {interval} 中发现 {len(longest_transcripts)} 个转录本")
            
        #     # 创建基因信息表
        #     gene_info = []
        #     for transcript in longest_transcripts:
        #         gene_info.append({
        #             '基因符号': getattr(transcript, 'gene_name', 'N/A'),
        #             '转录本ID': getattr(transcript, 'transcript_id', 'N/A'),
        #             '染色体': getattr(transcript, 'chromosome', 'N/A'),
        #             '起始位置': getattr(transcript, 'start', 'N/A'),
        #             '结束位置': getattr(transcript, 'end', 'N/A'),
        #             '链向': getattr(transcript, 'strand', 'N/A'),
        #         })
            
        #     gene_df = pd.DataFrame(gene_info)
        #     st.dataframe(gene_df, use_container_width=True)
        # else:
        #     st.info("在指定区间内未发现蛋白质编码基因")
        return longest_transcripts
            
    except Exception as e:
        st.warning(f"无法显示基因注释: {str(e)}")

if __name__ == "__main__":
    main() 