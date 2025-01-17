# set configuration
task="digit"

# set paths
path_cwd=$PWD
#path_data="/home/becker/repositories/AudioMNIST/data/" #"$path_repo/data_preprocessed/datasets_lmdb" #/foolset"
path_data="C:/Users/trist/PycharmProjects/AudioMNIST/AudioMNIST/preprocessed_data/"
timestamp=$(date +"%Y-%m-%d-%T")


# There are five splits for digit classification but only four for gender classification
if [ "$task" == "digit" ]; then
    nSplits=5
else
    nSplits=4
fi

for ((splitIdx=0;splitIdx<$nSplits;splitIdx+=1)); 
do
    # create folder for this run
    config=$path_cwd"/runs/"$timestamp"_"$task"_split"$splitIdx
    mkdir -p "$config"

    # set path to data for this run
    path_hdf5_txt="$path_data"AlexNet_"$task"_"$splitIdx"

    cp "$path_cwd/alexnet_solver.prototxt" "$config/alexnet_solver.prototxt"
    cp "$path_cwd/alexnet_train.prototxt" "$config/alexnet_train.prototxt"
    cp "$path_cwd/alexnet_test.prototxt" "$config/alexnet_test.prototxt"

    # create folder for snapshots
    mkdir -p "$config/snapshots"

    # adapt paths in solver
    sed -i "s!<path_snapshots>!$config/snapshots/AlexNet!g" "$config/alexnet_solver.prototxt"
    sed -i "s!<path_to_network.prototxt>!$config/alexnet_train.prototxt!g" "$config/alexnet_solver.prototxt"


    # adapt data paths in model
    sed -i "s!<path_to_training_data>!"$path_hdf5_txt"_train.txt!g" "$config/alexnet_train.prototxt"
    sed -i "s!<path_to_validation_data>!"$path_hdf5_txt"_validate.txt!g" "$config/alexnet_train.prototxt"
    sed -i "s!<path_to_test_data>!"$path_hdf5_txt"_test.txt!g" "$config/alexnet_test.prototxt"

    
    # task dependent adaptations
    if [ "$task" == "digit" ]; then
        nTestIter=60 # validation and test set each contain 12 vps -> 6000 samples
        sed -i "s!<num_output>!10!g" "$config/alexnet_train.prototxt"
        sed -i "s!<num_output>!10!g" "$config/alexnet_test.prototxt"
        # label of current task
        sed -i "s!<silent_label>!label_gender!g" "$config/alexnet_train.prototxt"
        sed -i "s!<silent_label>!label_gender!g" "$config/alexnet_test.prototxt"
        # unused label
        sed -i "s!<nonsilent_label>!label_digit!g" "$config/alexnet_train.prototxt"
        sed -i "s!<nonsilent_label>!label_digit!g" "$config/alexnet_test.prototxt"

    else
        nTestIter=30 # validation and test set each contains 6 vps -> 3000 samples
        sed -i "s!<num_output>!2!g" "$config/alexnet_train.prototxt"
        sed -i "s!<num_output>!2!g" "$config/alexnet_test.prototxt"
        # label of current task
        sed -i "s!<silent_label>!label_digit!g" "$config/alexnet_train.prototxt"
        sed -i "s!<silent_label>!label_digit!g" "$config/alexnet_test.prototxt"
        # unused label
        sed -i "s!<nonsilent_label>!label_gender!g" "$config/alexnet_train.prototxt"
        sed -i "s!<nonsilent_label>!label_gender!g" "$config/alexnet_test.prototxt"
    fi
    sed -i "s!<test_iter>!$nTestIter!g" "$config/alexnet_solver.prototxt"

    #TODO: make path to caffe less user specific

    echo "starting training"
#/home/becker/caffe/caffe-1.0/build/tools/caffe train -gpu "0" \

    C:/Users/trist/Desktop/caffe-windows-installation-tutorial/python/caffe train -gpu "0" \
                                                         -solver="$config/alexnet_solver.prototxt" \
                                                         2>&1 | tee "$config/alexnet_"$task"_split"$splitIdx"_train.log"
                                                        -model "$config/alexnet_test.prototxt" \
#                                                         -solver="$config/alexnet_solver.prototxt" \
#                                                         2>&1 | tee "$config/alexnet_"$task"_split"$splitIdx"_train.log"

    echo "starting testing"
    C:/Users/trist/Desktop/caffe-windows-installation-tutorial/python/caffe test -gpu "0" \
                                                        -iterations "$nTestIter" \
                                                        -weights "$config/snapshots/AlexNet_iter_10000.caffemodel" \
                                                        2>&1 | tee "$config/alexnet_"$task"_split_"$splitIdx"_test.log"


done