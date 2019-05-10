from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

def f(x):
	res = np.zeros(len(x),)
	for i in range(1, 30, 2):
		res += np.sin(i * x)/i
	return res

start=-20 #输入需要绘制的起始值（从左到右）
stop=20 #输入需要绘制的终点值
step=0.001#输入步长

x = np.arange(start,stop,step)
y = f(x)

print(x)
print(y)

fig=plt.figure(figsize=(6,6))#建立一个对象并设置窗体的大小，使其为正方形，好看
#注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）

plt.plot(x, y,label='First Curve')#在当前的对象上进行操作

plt.grid(False)#显示网格

plt.axis("equal")#设置了x、y刻度长度一致#需要放在x、ylim指令前
plt.xlim((min(x), max(x)))#显示的x的范围（不设置则由程序自动设置）
plt.ylim((min(y), max(y)))#显示的y的范围

plt.plot([min(x), max(x)], [0,0],label='x-axis')#用定义域最长距离的两倍作出x轴
plt.plot([0,0],[2*min(y),2*max(y)],label='y-axis')#用值域最长距离的两倍作出y轴
plt.legend()#显示旁注#注意：不会显示后来再定义的旁注
plt.show(fig)#没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()
