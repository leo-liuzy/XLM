for pair in ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh;
do
    echo $pair
    ./get-data-para.sh $pair

done
