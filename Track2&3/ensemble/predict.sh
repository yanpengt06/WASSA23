


# cd /users10/zjli/workspace/WASSA/task2\&3/
# # 预测结果
# python inference.py --task=MT_dis_emp --plm_model_num=0 --model_name=PLMMultiTaskRegressor --batch_size=8 --num_labels=1 --test_split=test

# python inference.py --task=emotion --plm_model_num=1 --model_name=PLMLSTMRegressor --batch_size=8 --num_labels=8 --test_split=test

# python inference.py --task=distress --plm_model_num=0 --model_name=PLMRegressor --batch_size=8 --num_labels=1 --test_split=test

# python inference.py --task=empathy --plm_model_num=0 --model_name=PLMRegressor --batch_size=8 --num_labels=1 --test_split=test

cd /users10/zjli/workspace/WASSA/ensemble/

python main.py --split test

