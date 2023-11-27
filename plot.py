import matplotlib.pyplot as plt

# 读取数据文件
data_file = 'nohup.out'
contrastive_loss = []
caption_loss = []
time = 0

with open(data_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('contrastive_loss:'):
            contrastive_loss.append(float(line.split(':')[1]))
        elif line.startswith('caption_loss:'):
            caption_loss.append(float(line.split(':')[1]))

# x轴数据
x = range(1, len(contrastive_loss) + 1)

# 绘制折线图
plt.plot(x, contrastive_loss, label='contrastive_loss')
plt.plot(x, caption_loss, label='caption_loss')

# 添加标题和轴标签
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 添加图例
plt.legend()

# 保存为文件
plt.savefig('13.png')

# 显示图形（可选）
# plt.show()