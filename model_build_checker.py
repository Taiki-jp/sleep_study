from nn.my_model import MyInceptionAndAttention, build_my_model
from pre_process.utils import FindsDir

fd = FindsDir("sleep")

model = build_my_model(input_shape=(512, 128, 1),
                       model = MyInceptionAndAttention(5, 512, 128, fd))

print(model.summary())