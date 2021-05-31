# SECTION : ライブラリのインポート
from utils import Utils
from my_setting import *
SetsPath().set()
from tanita_reader import TanitaReader
from psg_reader import PsgReader
from subjects_list import SubjectsList
from create_data import CreateData
from tqdm import tqdm

# SECTION : クラスオブジェクトの作成

m_findsDir = FindsDir("sleep")
m_subjectsList = SubjectsList()
m_createData = CreateData()
m_utils = Utils()

# SECTION :　各被験者ループ

for _, name in enumerate(tqdm(m_subjectsList.name_list)):
    records = list()
    m_tanitaReader = TanitaReader(name)
    m_psgReader = PsgReader(name)
    m_tanitaReader.readCsv()
    m_psgReader.readCsv()
    records.append(m_createData.makeSpectrum(m_tanitaReader.df, m_psgReader.df, kernel_size=1024, stride=16))
    m_utils.dumpWithPickle(records, m_findsDir, name)

