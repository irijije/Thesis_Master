class Config:    
    NAME = "test"
    TIMESTEP = 30
    N_FEATURES = 30

    FILENAME = "dataset/1_Final/Fin_host_session_submit_S.csv"
    DATAPATH = f"data/test/"

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 10

    #MODELNAME = f"models/cnn_id.h5"
    MODELNAME = f"models/cnn_{TIMESTEP}.h5"
    #MODELNAME = ['models/encoder.h5', 'models/decoder.h5']