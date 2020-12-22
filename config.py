# config for preprocess
import warnings
class DefaultConfig(object):
    '''
    一些参数定义及设置
    '''
    # raw data
    path_user = './data/raw_data/table1_user.txt'
    path_job = './data/raw_data/table2_jd.txt'
    path_action = './data/raw_data/table3_action.txt'
    # process data
    feature2idx_path = './data/process_data/feature2idx.txt'

    # the train\valid\test data (user_id,job_id,label)
    splitdata_path = './data/train_data'
    # processed df_user,df_job, for albertmodel
    user_processed_path = './data/train_data/processed_user.csv'
    job_processed_path = './data/train_data/processed_job.csv'

    # user continuous and category features
    user_co_cols = ['desire_jd_salary_id', 'cur_salary_id', 'cur_degree_id', 'birthday', 'work_years']
    user_ca_num_cols = ['live_city_id', 'desire_city_1', 'desire_city_2', 'desire_city_3']
    user_ca_text_cols = ['cur_industry_id', 'desire_industry_1', 'desire_industry_2', 'desire_industry_3',
                         'desire_jd_type_1', 'desire_jd_type_2', 'desire_jd_type_3']  # 'cur_jd_type'存在大量空值，所以舍弃
    user_exp_col = ['experience']
    # job continuous and category features
    job_co_cols = ['is_travel', 'require_nums', 'min_years', 'min_edu_level', 'max_salary', 'min_salary']
    job_ca_num_cols = ['city']
    job_ca_text_cols = ['jd_title', 'jd_sub_type']
    job_jd_col = ['job_description']


    # user和job合并后的cols，其中'user_id', 'jd_no',需要手动添加
    # 前面10co_cols 有大小关系，city属于类别，industry和jd_type是字符串，experience和job_description是短文本
    cols = ['require_nums',
            'birthday',
            'is_travel',
            'work_years', 'min_years',
            'cur_degree_id', 'min_edu_level',
            'desire_jd_salary_id', 'cur_salary_id', 'max_salary', 'min_salary',
            'live_city_id', 'desire_city_1', 'desire_city_2', 'desire_city_3', 'city',
            'cur_industry_id','desire_industry_1', 'desire_industry_2', 'desire_industry_3', 'jd_title',
            'desire_jd_type_1', 'desire_jd_type_2', 'desire_jd_type_3', 'jd_sub_type',
            'experience', 'job_description'
            ]
    # 用于local_math_net,相关的特征领域Embedding输入到局部匹配网络
    co_field_idx = [(3,4,4),(5,6,6),(7,9,9),(7,9,10)]
    ca_field_idx = [(0,4,4),(5,8,8)]
    # ALBERT
    albertpath = './ALBERT/'
    cat_token_len = 20
    experience_token_len = 300
    jd_token_len = 512
    # nn.Embedding for continuous and category featrues
    co_idx = 6 # continuous features have 6 kinds
    # ca_idx = 500 # city 编码数 # 修改为从本地dicts读取len(dicts['city'])
    co_num = len(user_co_cols) + len(job_co_cols) # 11 continuous features
    ca_num = len(user_ca_text_cols) + len(job_ca_text_cols)
    embdsize = 32 # 这是非文本特征向量维度，包含continuous 和category features，之前设置为16，
    albert_size = 312 # 这是albert模型产生的向量维度
    hiden_size = [512,128,64]
    out_size = 64
    local_dropout = 0.3
    num_heads = 2
    dropout = 0.4

    # for  deepFM model
    # ************************************************************************************************
    # user continuous and category features
    fmuser_co_cols = ['desire_jd_salary_id', 'cur_salary_id', 'cur_degree_id', 'birthday', 'work_years', ]
    fmuser_ca_cols = ['live_city_id', 'desire_city_1', 'desire_city_2', 'desire_city_3',
                    'cur_industry_id', 'desire_industry_1', 'desire_industry_2', 'desire_industry_3',
                    'cur_jd_type', 'desire_jd_type_1', 'desire_jd_type_2', 'desire_jd_type_3']
    # job continuous and category features
    fmjob_co_cols = ['require_nums', 'max_salary', 'min_salary', 'min_years', 'min_edu_level']
    fmjob_ca_cols = ['is_travel', 'city', 'jd_sub_type', 'industry']

    fmco_cols = ['desire_jd_salary_id', 'cur_salary_id', 'max_salary', 'min_salary',
               'cur_degree_id', 'min_edu_level',
               'work_years',  'min_years',
               'birthday','require_nums']
    fmca_cols = ['live_city_id', 'desire_city_1', 'desire_city_2', 'desire_city_3','city',
               'cur_industry_id', 'desire_industry_1', 'desire_industry_2', 'desire_industry_3', 'industry',
               'cur_jd_type', 'desire_jd_type_1', 'desire_jd_type_2', 'desire_jd_type_3', 'jd_sub_type',
               'is_travel']
    # for deepfm inputdata idx
    user_num_path = 'data/train_data/deepfm/user_idx.csv'
    job_num_path = 'data/train_data/deepfm/job_idx.csv'

    fmco_num = len(fmco_cols)
    fmca_num = len(fmca_cols)
    # ca_feat_size = [436, 58, 895, 4] # 修改为load_dicts后获取长度
    embedding_size = 10
    # ************************************************************************************************

    # for PJFNN and jrmpmm model
    # ************************************************************************************************
    stopwords_path = "./data/process_data/hit_stopwords.txt"
    word2idx_path = './data/process_data/word2idx.txt'
    idx2word_path = './data/process_data/idx2word.txt'
    user_idx_path = 'data/train_data/pjfnn/resume.sent.id'
    job_idx_path = 'data/train_data/pjfnn/job.sent.id'
    # words_count = 125586 # 修改为pjfnnmodel.py文件中的get_words_count()
    dim = 312
    hidden = 156
    sent_num = {"resume": 1,"job": 1}
    word_num = {"resume": 300,"job": 512}
    train_path = './data/train_data/pjfnn'
    # ************************************************************************************************

    # for single text model
    # ************************************************************************************************
    singletext_save_path = './model_checkpoints/singletext'
    # ************************************************************************************************

    # for single industry model
    # ************************************************************************************************
    industry_num = 5
    singleindustry_save_path = './model_checkpoints/singleindustry'
    # ************************************************************************************************

    # for single type model
    # ************************************************************************************************
    type_num = 4
    singletype_save_path = './model_checkpoints/singletype'
    # ************************************************************************************************

    # for single city model
    # ************************************************************************************************
    city_num = 5
    singlecity_save_path = './model_checkpoints/singlecity'
    # ************************************************************************************************

    # for single salary model
    # ************************************************************************************************
    salary_num = 5
    # ************************************************************************************************

    batch_size = {'train':36,'valid':36,'test':36}
    num_workers = {'train':2,'valid':2,'test':2}

    lr = 1e-4
    max_epoch = 15
    end_epoch = 5
    threshold = 0.1
    albert_save_path = './model_checkpoints/albert'
    muffin_save_path = './model_checkpoints/muffin'
    muffin_structure_save_path = './model_checkpoints/muffin_structure'
    pjfnn_save_path = './model_checkpoints/pjfnn'
    pjfff_save_path = './model_checkpoints/pjfff'
    jrmpm_save_path = './model_checkpoints/jrmpm'
    deepfm_save_path = './model_checkpoints/deepfm'

    train_size = 0.8
    valid_size = 0.1
    test_size = 0.1
    satisfied_num = 5
    satisfied_ratio = 0.65
    
    industries = {
        '房地产|建筑业':['房地产','地产','园林','建筑','造价','施工','装潢','工程资料','工程监理','工程总监','园林','工程技术','项目招投标','质量检验员','市政','幕墙','园艺','岩土工程','土建','硬装设计师','软装设计','管道工程技术','城市规划','港口工程技术','普工','物料','组装工','材料工程师','木工','技工','钢筋工'],
        '教育培训/科研':['校长','教授','教练','老师','教学','幼教','教师','培训','翻译','家教','教育','科研','学术','研究员','饮料研发','化学分析','化工研发','化妆品研发','化学技术应用','化工涂料研发','饲料研发','化验师','化学制剂研发'],
        '生活服务':['海外游','客服','客户咨询热线','保安','门卫','物业','旅游顾问','旅游','餐','导游','月嫂','保姆','保洁','厨','安检员','茶艺师','西点师','插花设计师','美容整形师','服务员','电工','化妆师','美容师','美容顾问','客房管理','营养师','空调工','按摩','监控维护','家政人员','生鲜食品加工','酒店试睡员'],
        '专业服务':['人力资源','招聘','行政','出纳员','财务','审计','前台','代驾','司机','文员','后勤人员','内勤人员','文案策划','猎头','活动执行','活动策划','市场策划','摄影师','广告文案策划','薪酬','品牌经理','员工关系','收银员','成本经理','内容运营','法务','律师','法务','纺织品设计','主持人','咨询师','促销主管','绘画','销售','咨询顾问','咨询经理','咨询项目管理','咨询总监','采购','项目经理','客户服务','质量管理','助理','总裁','总经理','秘书','运营','营运','合同管理','税务','市场主管','文档','店长','清算人员','市场经理','分销经理','质量检验员','酒店管理','风险管理','售后','市场营销','招商','代理','商务经理','安全管理','新店开发','风险控制','总监','代表处负责人','事业部管理','前厅接待','专业顾问','保险','理赔','主管','生产文员'],
        '批发/零售/贸易':['贸易','进出口','海关事务管理'],
        '制造业':['制造','数控','模具','钳工','机修','机械','维修','汽车质量管理','电焊工','冲床工','发动机','仪表工','汽车底盘','汽车零部件设计师','工业工程师','焊接工程师','车身设计工程师','汽车动力系统工程师','气动工程师','铸造'],
        '卫生及社会工作':['医师 ','医生','护士','药','医疗','兽医','理疗师','针灸','验光师'],
        '文化/体育/娱乐':['媒体','编辑出版','影视策划','媒介策划','作家','品牌策划','美术','会展策划','广告','总编','庆典策划','导演','文案策划','演员','音效师','印刷','后期制作','文字编辑','图书管理员','记者','经纪人','视频主播','配音员','陈列设计','游戏策划','排版设计','语音','放映员','灯光师','艺术指导','视频工程师'],
        '公共管理/社会保障':['政府','社会','党工团干事','储备干部','公务员'],
        '能源/环保/矿产':['弱电','环保技术工程师','机电','环境','电力','热能','电气','能源','处理工程师','水电工程师','天然气','采矿','光伏','生态治理','火力工程师','通信电源工程师','变压器'],
        '交通运输/仓储/物流':['物流','列车','快递员','乘务','轨道','运输经理','陆运','交通','船员','船舶驾驶','供应链','仓库','理货','地勤人员','搬运工','铲车'],
        '农林牧渔':['农','林业','养殖','畜牧师'],
        '互联网/IT/电子/通信':['用户界面（UI）设计','平面设计','销售数据分析','IT','数据分析师','数据运营','软件','开发工程师','系统架构设计师','计算机','电子','需求工程师','多媒体','编程','算法工程师','研发工程师','界面设计','系统集成工程师','信工程师','技术研发工程师','互联网','网络','前端开发','电脑','3D设计','CAD设计','数据库管理员','系统测试','硬件开发','信息技术','硬件测试','集成电路'],
        '金融业':['金融','银行','投资','证券','资金','基金','信托','行长','信贷','融资','信审核查','股票','买手','担保']
    }

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: config has not attribut %s" % k)
            setattr(self, k, v)

    def _print(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
        print('config end', '--' * 30)