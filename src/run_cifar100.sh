
BATCH=64
RATIO=3
EXEMPLAR_SIZE=2000
NUM_SPLITS=1
WEIGHT_DECAY=0.0001
EPOCHS=160

NUM_TASKS=10
NUM_FIRST_TASK=10

SEED_NUM=0
GPU_NUM=0

# ####### Using Batch Normalization #########

## Algorithm: Finetuning
EXP_NAME="CustomtBN_batch_size_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'finetuning' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --custom-split-bn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK 

## Algorithm: eeil
EXP_NAME="CustomtBN_batch_size_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'eeil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --custom-split-bn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK 

## Algorithm: lucir
EXP_NAME="CustomtBN_batch_size_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'lucir' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --custom-split-bn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK 

## Algorithm: ssil
EXP_NAME="CustomtBN_batch_size_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'ssil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --custom-split-bn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK 




# ####### Using Continual Normalization #########

CN_NUM=8

## Algorithm: Finetuning
EXP_NAME="ContinualBN_num_groups_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'finetuning' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --cn --cn-num $CN_NUM --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

## Algorithm: eeil
EXP_NAME="ContinualBN_num_groups_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'eeil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs 1  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --cn --cn-num $CN_NUM --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

## Algorithm: lucir
EXP_NAME="ContinualBN_num_groups_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'lucir' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --cn --cn-num $CN_NUM --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

## Algorithm: ssil
EXP_NAME="ContinualBN_num_groups_${BATCH}_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'ssil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --cn --cn-num $CN_NUM --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  




# ####### Using TBBN #########

## Algorithm: Finetuning
EXP_NAME="TBBN_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'finetuning' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --tbbn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

## Algorithm: eeil
EXP_NAME="TBBN_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'eeil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --tbbn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

## Algorithm: lucir
EXP_NAME="TBBN_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'lucir' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --tbbn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

# Algorithm: ssil
EXP_NAME="TBBN_ratio_${RATIO}_exemplar_size_${EXEMPLAR_SIZE}_resnet18_seed_${SEED_NUM}"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_incremental.py --gpu 0 --seed $SEED_NUM --approach 'ssil' --datasets 'cifar100' --network 'resnet32' --batch-size $BATCH --num-tasks $NUM_TASKS --nepochs $EPOCHS  --num-exemplars $EXEMPLAR_SIZE --lr-factor 10 --weight-decay $WEIGHT_DECAY --exemplar-selection herding --num-workers 8 --exp-name $EXP_NAME --tbbn --batch-ratio $RATIO --nc-first-task $NUM_FIRST_TASK  

