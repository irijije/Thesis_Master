class Config:    
    N_ID = 2048
    
    MAX_TIMESTEP = 300
    UNIT_TIMESTEP = 30
    N_INTVL = int(MAX_TIMESTEP/UNIT_TIMESTEP)
    N_FEATURES = 30

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPOCHS = 10

    FILENAME = "dataset/0_Preliminary/1_Submission/Pre_submit_D.csv"
    #FILENAME = "dataset/1_Final/Fin_host_session_submit_S.csv"
    DATAPATH = f"data/hand/"
    #DATAPATH = f"data/raw/"

    #MODEL_NAME = "models/cnn_id.h5"
    MODEL_NAME = "models/cnn_hand.h5"
    #MODEL_NAME = "models/cnn_mk.h5"
    #MODEL_NAME = "models/cnn_lstm.h5"
    #MODEL_NAME = "models/cnn_raw"
    #MODEL_NAME = ['models/encoder_mt.h5', 'models/decoder_mt.h5']