import tensorflow as tf
import os
import sys
ROOT_PATH='/home/data/dl_learning/tf_car_license_dataset'
train_path=os.path.join(ROOT_PATH,'train_images/training-set/')
validate_path=os.path.join(ROOT_PATH,'train_images/validation-set/')
test_path=os.path.join(ROOT_PATH,'test_images/')
num_pair_label=[]
letter_pair_label=[]
zh_label=['京','闽','粤','苏','沪','浙']
zh_pair_label=[]
BATCH_SIZE=10
INPUT_SIZE=32*40

sess = tf.InteractiveSession()
tf.train.start_queue_runners()
for i in range(0,10):
    num_pair_label.append((i,i))
for i in range(0,26):
    letter_pair_label.append((chr(ord('A')+i),i+10))
for i,zh in enumerate(zh_label):
    zh_pair_label.append((zh,i+36))
total_pair_label=[num_pair_label,letter_pair_label,zh_pair_label]

def read_image(path,batch_size):
    image_path=[]
    labels=[]
    classes=sorted(os.walk(path).__next__()[1])
    for c in classes:
       # print(c)
        label=float(str(c))
        one_hot_label=[]
        for i in range(41):
            if i==label:
                one_hot_label.append(1)
            else:
                one_hot_label.append(0)
        c_dir=os.path.join(path,c)
        walk=os.walk(c_dir).__next__()[2]
        for sample in walk:
            if sample.endswith('.bmp'):
                image_path.append(os.path.join(c_dir,sample))
                labels.append(one_hot_label)
    #print(labels)
   # sys.exit(0)
    image_path=tf.convert_to_tensor(image_path,tf.string)
    labels=tf.convert_to_tensor(labels,tf.float32)
    #print(labels.shape)
    image_path,label=tf.train.slice_input_producer([image_path,labels],shuffle=True)
   # print("image,label:",image_path.shape,label.shape)
    image=tf.read_file(image_path)
    image=tf.image.decode_bmp(image,channels=1)
    image=tf.image.resize_images(image,size=[32,40])
    image=image*0.1/127.5-1.0
    X,Y=tf.train.batch([image,label],batch_size=batch_size,num_threads=4,capacity=batch_size*8)
    return X,Y

                                        
X,Y=read_image(train_path,100)
#print(X,Y)
X_t,Y_t=read_image(validate_path,1)



def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    weight=tf.Variable(initial_value=initial)
    return weight

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    bias=tf.Variable(initial_value=initial)
    return bias

def conv_op(in_tensor,kernel,strides=[1,1,1,1],padding='SAME'):
    conv_out=tf.nn.conv2d(in_tensor,kernel,strides=strides,padding=padding)
    return conv_out
def max_pool_2X2(in_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
    max_pool=tf.nn.max_pool(in_tensor,ksize,strides,padding)
    return max_pool
def simple_cnn():
    x=tf.placeholder(tf.float32,[None,32,40,1])
    y_=tf.placeholder(tf.float32,[None,41])
    x_image=tf.reshape(x,[-1,32,40,1])

    w1 = [5,5,1,32]
    b1 = [32]
    w2 = [5,5,32,64]
    b2 = [64]
    wfc1 = [8*10*64,1024]
    bfc1 = [1024]
    wfc2 = [1024,41]
    bfc2 = [41]
    #layer 1
    W_conv1=weight_variable(w1)
    b_conv1=bias_variable(b1)
    h_conv1=tf.nn.relu(conv_op(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2X2(h_conv1)
    #layer 2
    W_conv2=weight_variable(w2)
    b_conv2=bias_variable(b2)
    h_conv2=tf.nn.relu(conv_op(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_2X2(h_conv2)
    #full connect layer 1
    h_pool2_flat=tf.reshape(h_pool2,[-1,8*10*64])
    W_fc1=weight_variable(wfc1)
    b_fc1=bias_variable(bfc1)
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    #drop_out
    keep_prob=tf.placeholder(tf.float32)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    #fc2
    W_fc2=weight_variable(wfc2)
    b_fc2=bias_variable(bfc2)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    #loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 #   print(y_conv.shape,y_.shape)
    #import sys
   # sys.exit(0)
    # estimate accuarcy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    tf.global_variables_initializer().run()
    

    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    epoch = 0
    try:
        while not coord.should_stop():
            # 获取训练用的每一个batch中batch_size个样本和标签
            data, label = sess.run([X, Y])
            test_d,test_l=sess.run([X_t,Y_t])
            #          import sys
   #         print(data.shape,label.shape)
            
            #abel=tf.reshape(label,[None,1])
            #print(label.shape)
            if epoch%100==0:
                train_accuracy=accuracy.eval(feed_dict={x:data,y_:label,keep_prob:1.0})
                print('step %d, training accuarcy %g'%(epoch, train_accuracy))
                test_acc=accuracy.eval(feed_dict={x:test_d,y_:test_l,keep_prob:1.0})
                print("step %d,testing accuracy %g"%(epoch,test_acc))
                #print(cross_entropy)
            train_step.run(feed_dict={x:data,y_:label,keep_prob:0.5}) 
            #print(correct,label)
            epoch = epoch + 1
    except tf.errors.OutOfRangeError:  # num_epochs 次数用完会抛出此异常
        print("---Train end---")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('---Programm end---')
        
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束 
    #print("test accuracy %g"%accuarcy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
    return 

if __name__=='__main__':
    simple_cnn()
    



