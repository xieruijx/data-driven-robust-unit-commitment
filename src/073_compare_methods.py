# Organize and compare the results of different methods
from utils.io import IO

index_u_l_predict = 0
type_u_l = 'test'

IO().organize_methods(30, index_u_l_predict, type_u_l, 0.05, folder_outputs='./results/outputs/')
