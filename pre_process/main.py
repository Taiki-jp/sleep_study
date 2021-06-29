from pre_process.file_reader import FileReader
from data_analysis.utils import Utils
from pre_process.tanita_reader import TanitaReader
from pre_process.psg_reader import PsgReader
from pre_process.create_data import CreateData
from tqdm import tqdm
from data_analysis.py_color import PyColor
import datetime
import sys

# ハイパーパラメータの設定
DATA_TYPE = "spectrum"
date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
FIT_POS = "top"

# オブジェクトの作成
CD = CreateData()
FR = FileReader(n_class=5)  # TODO : ここにクラス数を指定する必要がないけど、FileReaderの実装上必要になっている
FR.sl.sets_filename(data_type=DATA_TYPE, n_class=5)  # TODO : 同上
utils = Utils(file_reader=FR)

for name in FR.sl.name_list:
    print(PyColor.GREEN,
          PyColor.FLASH,
          f" *** {name} を開始します ***",
          PyColor.END)
    records = list()
    tanita = TanitaReader(name)
    psg = PsgReader(name)
    tanita.readCsv()
    psg.readCsv()
    # 始まりの時刻がそろっていることをassert
    try:
          assert tanita.df["time"][0] == psg.df["time"][0]
    except:
          print("starting time is not corrected")
          sys.exit(1)
    records.append(CD.makeSpectrum(tanita.df, 
                                   psg.df, 
                                   kernel_size=1024, 
                                   stride=16,
                                   fit_pos=FIT_POS))
    utils.dump_with_pickle(records, name, data_type=DATA_TYPE,
                           fit_pos = FIT_POS)
