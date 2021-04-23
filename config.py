class Config:    
    NAME = "test"
    TIMESTEP = 30
    N_FEATURES = 30

    FILENAME = "dataset/0_Preliminary/0_Training/Pre_train_D_1.csv"
    #FILENAME = "dataset/1_Final/Fin_host_session_submit_S.csv"
    DATAPATH = f"data/test/"

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 100

    #MODELNAME = f"models/{TIMESTEP}.h5"
    MODELNAME = ['models/encoder.h5', 'models/decoder.h5']