#!/bin/bash


for x in 1 5 4 3
do
    if [[ $x == 1 ]]; then
        exp=qalb14
    else
        exp=qalb14_${x}
    fi

        input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep/pnx/$exp
        output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_mt/pnx/$exp

        python get_binary_labels.py \
            --input $input_dir/train_pnx.txt \
            --output $output_dir/train_pnx.bin.txt


            cat $output_dir/train_pnx.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
            cat $output_dir/train_pnx.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

            sed -i '1d' $output_dir/labels.txt
            sed -i '1d' $output_dir/labels.bin.txt



        input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep/nopnx/$exp
        output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_mt/nopnx/$exp

        python get_binary_labels.py \
            --input $input_dir/train_nopnx.txt \
            --output $output_dir/train_nopnx.bin.txt


            cat $output_dir/train_nopnx.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
            cat $output_dir/train_nopnx.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

            sed -i '1d' $output_dir/labels.txt
            sed -i '1d' $output_dir/labels.bin.txt


done


# for x in 1 5 4 3
# do
#     if [[ $x == 1 ]]; then
#         exp=qalb14
#     else
#         exp=qalb14_${x}
#     fi

#     for i in 10 20 30
#     do
#         input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_prune_${i}/pnx/$exp
#         output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_prune_${i}_mt/pnx/$exp

#         python get_binary_labels.py \
#             --input $input_dir/train_pnx.txt \
#             --output $output_dir/train_pnx.bin.txt


#             cat $output_dir/train_pnx.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
#             cat $output_dir/train_pnx.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

#             sed -i '1d' $output_dir/labels.txt
#             sed -i '1d' $output_dir/labels.bin.txt



#         input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_prune_${i}/nopnx/$exp
#         output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/pnx_sep_prune_${i}_mt/nopnx/$exp

#         python get_binary_labels.py \
#             --input $input_dir/train_nopnx.txt \
#             --output $output_dir/train_nopnx.bin.txt


#             cat $output_dir/train_nopnx.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
#             cat $output_dir/train_nopnx.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

#             sed -i '1d' $output_dir/labels.txt
#             sed -i '1d' $output_dir/labels.bin.txt


#     done

# done



# for x in 1 5 4 3
# do
#     if [[ $x == 1 ]]; then
#         exp=qalb14
#     else
#         exp=qalb14_${x}
#     fi

#     for i in 10 20 30
#     do
#         input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/prune_${i}/$exp
#         output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/prune_${i}_mt/$exp

#         python get_binary_labels.py \
#             --input $input_dir/train.txt \
#             --output $output_dir/train.bin.txt


#             cat $output_dir/train.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
#             cat $output_dir/train.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

#             sed -i '1d' $output_dir/labels.txt
#             sed -i '1d' $output_dir/labels.bin.txt

#     done

# done




# for x in 1 5 4 3
# do
#     if [[ $x == 1 ]]; then
#         exp=qalb14
#     else
#         exp=qalb14_${x}
#     fi

#     for split in dev train test
#     do
#         input_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/$exp
#         output_dir=/scratch/ba63/arabic-text-editing/tagger_data_new/${exp}_mt
    
#         python get_binary_labels.py \
#             --input $input_dir/${split}.txt \
#             --output $output_dir/${split}.bin.txt

#             if [[ $split == "train" ]]; then

#                 cat $output_dir/${split}.bin.txt | cut -f2 | sort | uniq > $output_dir/labels.txt
#                 cat $output_dir/${split}.bin.txt | cut -f3 | sort | uniq > $output_dir/labels.bin.txt

#                 sed -i '1d' $output_dir/labels.txt
#                 sed -i '1d' $output_dir/labels.bin.txt
#             fi

#     done

# done
