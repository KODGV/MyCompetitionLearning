# IJCAI比赛

数据说明：https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.4cc6311f4ZhE5Y&raceId=231647&postsId=4601#pages%3D6



1.先用Pd.pivot画出透视图，分析day和hour对rate的关系

2.应该先分析转化率而不是分析corr()

2.数据嫁接的思想：

砧木：数据量大，但是实际价值不大

接穗：数据量少，但是实际价值大

映射：

砧木：前七天的数据，量大但是价值低

接穗：最后一个上午的数据，量少但是价值高

嫁接的思想：

利用前七天的预测得到概率特征，接在接穗最后一个上午的数据上面，而且要注意的是，亲缘关系越近，嫁接成功率越高，所以在这个问题上也是item相同品别的做这种“嫁接“操作效果才会好，而我们的初赛数据和复赛数据的item并不是一个品类的，所以价值不大。

前七天预测第八天（注意预测包含训练和测试集），拆上午做训练，下午做验证

![1544412592025](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1544412592025.png)



4.统计特征

5.时差特征

6.排序特征（用户与商品的第几次交互，倒数第几次交互）------倒数特征



7..发现每个人的平均次数较少，所以和通常的根据用户的历史数据预测未来是否购买问题是不一样的，没必要进行滑窗统计user_id的特征

采用的是：

1.对已有属性做词袋模型，即0/1变量，判断该用户的搜索属性占 已有属性的多少条(这样也会很多，所以只统计了前100的property)

2.根据groupbyuserid取属性的mean，统计出用户对于商品属性的占比

3.再根据groupbyitem取商品属性的占比的mean，统计出查看该商品的仁德不同属性的平均爱好

高大上叫做embeding(就是通过一个跳板来映射两个表之间的关系)

二.Sample Embedding
sample_emb_x=[x1,x2,x3,x4,...,xn]                   xn为第n个property在不在predict_category_property中
sample_emb_y=[y1,y2,y3,y4,...,yn]                  yn为第n个property在不在item_property_list中

一个user有很多个不同的item交互样本，一个item也有很多不同的user交互样本
user_emb_x=mean([sample_emb_x_1,sample_emb_x_2,...,sample_emb_x_k])         sample_emb_x_k为该user的第k条样本的sample_emb
user_emb_y=mean([sample_emb_y_1,sample_emb_y_2,...,sample_emb_y_k])         sample_emb_y_k为该user的第k条样本的sample_emb

通过这种对所有样本的sample_emb做mean操作来对user做embedding
item_emb_x=mean([user_emb_x_1,user_emb_x_2,...,user_emb_x_k])               user_emb_x_k为该item的第k条样本的user_emb
item_emb_y=mean([user_emb_y_1,user_emb_y_2,...,user_emb_y_k])               user_emb_y_k为该item的第k条样本的user_emb

通过这种对所有样本的use_emb做mean操作来对item做embedding
至此通过predict_category_property，item_property_list这两条信息对sample，user，item做了embedding。
得到了6*n个特征，n的大小视情况而定，这里我取了出现次数top100的property来做我的embedding，所以总共6*100个特征。 