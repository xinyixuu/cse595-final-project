
import matplotlib.pyplot as plt
import numpy as np

y = [0.6043, 0.5710  , 0.5773   , 0.5492  ,  0.5207      , 0.4996   , 0.4900]
x = ["baseline", "k=1", "k=2", "k=4", "k=8", "k=16", "k=32"]
colors = ["gray"] + ["skyblue"] * (len(x) - 1)

all_values = [
   [0.6201, 0.5964, 0.5889, 0.6147, 0.6014],     # baseline
   [0.5876, 0.5632, 0.5541, 0.5830, 0.5672],     # k=1
    [0.5924, 0.5685, 0.5609, 0.5943, 0.5710],     # k=2
    [0.5634, 0.5379, 0.5303, 0.5598, 0.5546],     # k=4
    [0.5358, 0.5094, 0.5021, 0.5370, 0.5193],     # k=8
    [0.5133, 0.4881, 0.4804, 0.5162, 0.4999],     # k=16
   [0.5041, 0.4783, 0.4712,0.5030, 0.4934]      # k=32
]


plt.figure()
bars = plt.bar(x, y, color=colors)
plt.ylabel("Accuracy")
plt.title("DEONTOLOGY: Accuracy with Different ICL Random Label(with MLP Head)")

plt.axhline(y=0.6043, linestyle="--", color="red")
plt.ylim(0.47, 0.63)


for i, vals in enumerate(all_values):
    cx = bars[i].get_x() + bars[i].get_width() / 2
    plt.scatter([cx] * len(vals), vals, s=5, color="purple", zorder=5)

plt.savefig("/home/guangchen_li/dev/Personal/final_project/pic/pic_2")
plt.show()