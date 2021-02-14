# VAT-BNN

### Usage

训练base：python base_vat.py --do_train --learning_rate=0.001 --label_num=xx --epsilon=xx

训练vat：python base_vat.py --do_train --learning_rate=0.001 --label_num=xx --epsilon=xx  --vat

其中epsilon控制攻击半径，label_num为标注数量，为0代表全标注

训练pi模型：python pi_model.py --do_train --label_num=xx

训练mi：python mi_vat.py --do_train --label_num=0 --pretrained_config=vat_batch-100_dataset-mnist_labelnum-0_epsilon-0.5 --epsilon=0.1

pretrained_config为预训练模型所在文件夹

### 感想
