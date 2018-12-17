# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

########################################################################### 回归
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
import os
# 别人的自定义损失函数,在parameter里面：object里面赋值
def custom_loss(y_true,y_pred):
    penalty=2.0
    grad=-y_true/y_pred+penalty*(1-y_true)/(1-y_pred) #梯度
    hess=y_true/(y_pred**2)+penalty*(1-y_true)/(1-y_pred)**2 #2阶导
    return grad,hess
# 自定义评价函数
def mse(y_pred,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
    labels=dtrain.get_label() #提取label
    t=mean_absolute_error(labels, y_pred)
    print(t)
    return 'mse',t


parameters = {'nthread':-1, # cpu 线程数 默认最大
              'objective':'reg:linear',#多分类or 回归的问题    若要自定义就替换为custom_loss（不带引号）
              'learning_rate': .01, #so called `eta` value 如同学习率
              'max_depth': 6,# 构建树的深度，越大越容易过拟合
              'min_child_weight': 4,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 1,#设置成1则没有运行信息输出，最好是设置为0.
              'subsample': 0.7, # 随机采样训练样本
              'colsample_bytree': 0.7,# 生成树时进行的列采样
              'n_estimators': 100,# 树的个数跟num_boost_round是一样的，所以可以设置无限大，靠early_stop
              'gamma':0.1,# 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
              'seed':1000 #随机种子
              #'alpha':0, # L1 正则项参数
              #'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
              #'num_class':10, # 类别数，多分类与 multisoftmax 并用
              }

def mergeToOne( X, X2):
    X3 = []
    for i in range(X.shape[0]):
        tmp = np.array([list(X[i]), list(X2[i])])
        X3.append(list(np.hstack(tmp)))
    X3 = np.array(X3)
    return X3
def get_XgbRegressor(train_data,train_target,test_data,feature_names,parameters,early_stopping_rounds,num_folds,eval_metric,model_name='model',stratified=False):
    '''
    :param train_data: 一定是numpy
    :param train_target:
    :param parameters:
    :param round:
    :param k:
    :param eval_metrics:自定义 or 内置字符串
    :return:
    '''
    reg=XGBRegressor()
    reg.set_params(**parameters)

    # 定义一些变量
    oof_preds = np.zeros((train_data.shape[0],))
    sub_preds = np.zeros((test_data.shape[0],))
    feature_importance_df = pd.DataFrame()
    cv_result = []

    # K-flod
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1234)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1234)
    X_train_newfeature=np.zeros((1,1))
    for n_flod, (train_index, val_index) in enumerate(folds.split(train_data, train_target)):
        train_X=train_data[train_index]
        val_X=train_data[val_index]
        train_Y=train_target[train_index]
        val_Y=train_target[val_index]
        # 参数初步定之后划分20%为验证集，准备一个watchlist 给train和validation set ,设置num_round 足够大（比如100000），以至于你能发现每一个round 的验证集预测结果，
        # 如果在某一个round后 validation set 的预测误差上升了，你就可以停止掉正在运行的程序了。
        watchlist= [(train_X, train_Y), (val_X, val_Y)]

        # early_stop 看validate的eval是否下降，这时候必须传eval_set,并取eval_set的最后一个作为validate
        reg.fit(train_X,train_Y,early_stopping_rounds=early_stopping_rounds, eval_set=watchlist,eval_metric=eval_metric)

        ## 生成gbdt新特征
        new_feature = reg.apply(val_X)
        if X_train_newfeature.shape[0]==1:
            X_train_newfeature=mergeToOne(val_X,new_feature)
        else:
            X_train_newfeature = mergeToOne(val_X,new_feature)
            X_train_newfeature=np.concatenate((X_train_newfeature,mergeToOne(new_feature, val_X)),axis=0)
        print (X_train_newfeature)
       # 获得每次的预测值补充
        oof_preds[val_index]=reg.predict(val_X)
        # 获得预测的平均值，这里直接加完再除m
        sub_preds+= reg.predict(test_data)
        result = mean_absolute_error(val_Y, reg.predict(val_X))
        print('Fold %2d macro-f1 : %.6f' % (n_flod + 1, result))
        cv_result.append(round(result,5))
        gc.collect()
        # 默认就是gain 如果要修改要再参数定义中修改importance_type
        # 保存特征重要度
        gain = reg.feature_importances_
        fold_importance_df = pd.DataFrame({'feature': feature_names,
                                           'gain': 100 * gain / gain.sum(),
                                           'fold': n_flod,
                                           }).sort_values('gain', ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # 进行保存
    sub_preds=sub_preds/folds.n_splits
    new_feature=reg.apply(test_data)
    X_test_newfeature = mergeToOne(test_data, new_feature)

    if not os.path.isdir('./sub'):
        os.makedirs('./sub')
    pd.DataFrame(oof_preds,columns=['class']).to_csv('./sub/val_{}.csv'.format(model_name), index=False)
    pd.DataFrame(sub_preds, columns=['class']).to_csv('./sub/test_{}.csv'.format(model_name), index=False)
    print('cv_result', cv_result)

    if not os.path.isdir('./gbdt_newfeature'):
        os.makedirs('./gbdt_newfeature')

    np.save("./gbdt_newfeature/train_newfeature.npy", X_train_newfeature)
    np.save("./gbdt_newfeature/test_newfeature.npy", X_test_newfeature)
    save_importances(feature_importance_df, model_name)
    return reg,sub_preds
def save_importances(feature_importance_df,model_name):
    if not os.path.isdir('./feature_importance'):
        os.makedirs('./feature_importance')
    ft = feature_importance_df[["feature", "gain"]].groupby("feature").mean().sort_values(by="gain",ascending=False)
    ft.to_csv('./feature_importance/importance_lightgbm_{}.csv'.format(model_name), index=True)






########################################################################################## 分类
from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import numpy as np
import os
import gc

clf_parameters = {'nthread':-1, # cpu 线程数 默认最大
              'objective':'multi:softmax',#多分类or 回归的问题    若要自定义就替换为custom_loss（不带引号）
              'learning_rate': .01, #so called `eta` value 如同学习率
              'max_depth': 6,# 构建树的深度，越大越容易过拟合
              'min_child_weight': 4,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 1,#设置成1则没有运行信息输出，最好是设置为0.
              'subsample': 0.7, # 随机采样训练样本
              'colsample_bytree': 0.7,# 生成树时进行的列采样
              'n_estimators': 500,# 树的个数跟num_boost_round是一样的，所以可以设置无限大，靠early_stop
              'gamma':0.1,# 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
              'seed':1000 #随机种子
              #'alpha':0, # L1 正则项参数
              #'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
              }

n_class=3













def clf_custom_loss(y_true,y_pred):
    penalty=2.0
    grad=-y_true/y_pred+penalty*(1-y_true)/(1-y_pred) #梯度
    hess=y_true/(y_pred**2)+penalty*(1-y_true)/(1-y_pred)**2 #2阶导
    return grad,hess

# 自定义评价函数
def clf_mse(y_pred,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
    labels=dtrain.get_label() #提取label
    ######### 分类预测的都是概率哦，所以这里要取一个max类别
    y_pred = np.argmax(y_pred.reshape(n_class, -1), axis=0)
    score=mean_absolute_error(labels, y_pred)
    return 'mse',score




# 分类的时候要注意！！！！！！！！
# k-flod的时候要按层次拿出来，有一个shuffler我这里就没实现了，否则预测的类别会出现变小甚至报错

def get_XgbClassifer(train_data,train_target,test_data,feature_names,parameters,early_stopping_rounds,num_folds,eval_metric,model_name='model',stratified=True):
    '''
    :param train_data: 一定是numpy
    :param train_target:
    :param parameters:
    :param round:
    :param k:
    :param eval_metrics:自定义 or 内置字符串
    :return:
    '''


    # 如果在param中设置，会莫名报参数不存在的错误
    clf=XGBClassifier(num_class=n_class)
    clf.set_params(**parameters)

    # 定义一些变量
    oof_preds = np.zeros((train_data.shape[0],n_class))
    sub_preds = np.zeros((test_data.shape[0],n_class))
    feature_importance_df = pd.DataFrame()
    cv_result = []

    # K-flod
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1234)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1234)
    for n_flod,(train_index, val_index) in enumerate(folds.split(train_data,train_target)):
        train_X=train_data[train_index]
        val_X=train_data[val_index]
        train_Y=train_target[train_index]
        val_Y=train_target[val_index]

        # 参数初步定之后划分20%为验证集，准备一个watchlist 给train和validation set ,设置num_round 足够大（比如100000），以至于你能发现每一个round 的验证集预测结果，
        # 如果在某一个round后 validation set 的预测误差上升了，你就可以停止掉正在运行的程序了。
        watchlist= [(train_X, train_Y)]
        # early_stop 看validate的eval是否下降，这时候必须传eval_set,并取eval_set的最后一个作为validate
        clf.fit(train_X,train_Y,early_stopping_rounds=early_stopping_rounds, eval_set=watchlist,eval_metric=eval_metric)
        # 获得每次的预测值补充
        oof_preds[val_index]=clf.predict_proba(val_X)
        # 获得预测的平均值，这里直接加完再除m
        sub_preds+= clf.predict_proba(test_data)
        # 计算当前准确率
        result=mean_absolute_error(val_Y,clf.predict(val_X))
        print('Fold %2d macro-f1 : %.6f' % (n_flod + 1, result))
        print(type(result))
        cv_result.append(round(result,5))
        gc.collect()



        # 默认就是gain 如果要修改要再参数定义中修改importance_type
        # 保存特征重要度
        gain = clf.feature_importances_
        fold_importance_df = pd.DataFrame({'feature':feature_names,
                                           'gain':100*gain/gain.sum(),
                                           'fold':n_flod,
                                           }).sort_values('gain',ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    # 进行保存
    sub_preds=sub_preds/folds.n_splits
    if not os.path.isdir('./cv'):
        os.makedirs('./cv')
    pd.DataFrame(oof_preds,columns=['class_'+ str(i) for i in range(n_class)]).to_csv('./cv/val_prob_{}.csv'.format(model_name), index= False, float_format = '%.4f')
    pd.DataFrame(sub_preds, columns=['class_' + str(i) for i in range(n_class)]).to_csv('./cv/test_prob_{}.csv'.format(model_name), index=False, float_format='%.4f')
    oof_preds = [np.argmax(x) for x in oof_preds]
    sub_preds = [np.argmax(x) for x in sub_preds]
    if not os.path.isdir('./sub'):
        os.makedirs('./sub')
    pd.DataFrame(oof_preds,columns=['class']).to_csv('./sub/val_{}.csv'.format(model_name), index=False)
    pd.DataFrame(sub_preds, columns=['class']).to_csv('./sub/test_{}.csv'.format(model_name), index=False)


    save_importances(feature_importance_df, model_name)
    return clf

def save_importances(feature_importance_df,model_name):
    if not os.path.isdir('./feature_importance'):
        os.makedirs('./feature_importance')
    ft = feature_importance_df[["feature", "gain"]].groupby("feature").mean().sort_values(by="gain",ascending=False)
    ft.to_csv('./feature_importance/importance_lightgbm_{}.csv'.format(model_name), index=True)

#######################进阶操作###############
# 可视化树结构
# xgb.plot_tree(xgbcl);、
# xgb.to_graphviz(xgbcl)
# # 特征重要度
# xgb.plot_importance(xgbcl);
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn import datasets
    from xgboost.sklearn import XGBClassifier
    house=datasets.load_boston()

    get_XgbRegressor(house.data,house.target,house.data,house.feature_names,parameters,100,2,mse)

    # irse=datasets.load_iris()
    # get_XgbClassifer(irse.data,irse.target,irse.data,['X1','X2','X3','X4'],clf_parameters,100,2,clf_mse)

