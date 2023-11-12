from enum import Enum
import numpy as np
from arc.interface import BoardPair, Board
import imutils
import scipy
import skimage

class Direction(Enum):
    horizontal = 1
    vertical = 2
    both = 3

def noop(o):
    return o

def cropInputAndOutput(inputPair:BoardPair, cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row)->BoardPair:
    input_np_board = inputPair.input.np
    output_np_board = inputPair.output.np
    input_np_board = cropBoard(input_np_board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row).np
    output_np_board = cropBoard(output_np_board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row).np

    return BoardPair(input=input_np_board.tolist(),output=output_np_board.tolist())
def cropInputOnly(inputPair:BoardPair, cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row)->BoardPair:
    input_np_board = inputPair.input.np
    input_np_board = cropBoard(input_np_board,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row).np

    return BoardPair(input=input_np_board.tolist(),output=inputPair.output)

def cropBoard(boardIn:np.ndarray,cols_to_rem_choice,rows_to_rem_choice,dir_choice_col,dir_choice_row)->Board:
    if cols_to_rem_choice > 0 :
        if dir_choice_col == -1:
            boardIn = boardIn[cols_to_rem_choice:,:]
        else:
            boardIn = boardIn[:-cols_to_rem_choice,:]
    if rows_to_rem_choice > 0:
        if dir_choice_row == -1:
            boardIn = boardIn[:,rows_to_rem_choice:]
        else:
            boardIn = boardIn[:,:-rows_to_rem_choice]

    return Board(__root__ = boardIn.tolist())

def floatRotateAll(board_pair:BoardPair,angle:int,superres_scale_fac)->BoardPair:
    board_pair.input = floatRotate(board_pair.input,angle,superres_scale_fac)
    board_pair.output = floatRotate(board_pair.output,angle,superres_scale_fac)
    return board_pair

def floatRotate(board_in:Board,angle:int,superres_scale_fac)->Board:
    #board_in = Board(__root__= np.flip(board_in.np, 1).tolist())
    board_in_np_border = np.pad(board_in.np,((1,1),(1,1)),constant_values = 0)
    board_in_np_border = Board(__root__ = board_in_np_border.tolist())
    superres_scale = 10
    upscaled_factor_2_board = superResolutionBoard(board_in_np_border,superres_scale,Direction.both)
    board_in_scaled_np = upscaled_factor_2_board.np
    board_in_separated = np.eye(10)[board_in_scaled_np]
    board_in_rotated = scipy.ndimage.rotate(board_in_separated, angle, order=0, prefilter=False)

    # convolve the board with 10x10 kernel of ones, stride of 10, so that we downscale the board by 10
    def func(block,axis):
        return np.mean(block,axis=axis)
        return np.mean(block[:,:,:,3:7,3:7,:],axis=axis)
    upscale_fac = superres_scale_fac
    downscaled_board = skimage.measure.block_reduce(board_in_rotated,\
            (superres_scale//upscale_fac,superres_scale//upscale_fac,1),\
            func = func)
    downscaled_board[:,:,0] = downscaled_board[:,:,0]/2.1
    ret = np.argmax(downscaled_board,-1)
    #ret = scipy.ndimage.rotate(board_in.np, angle, order=0, prefilter=False)
    return Board(__root__ = ret.tolist())


def doubleBoard(boardIn:Board,separation:int,is_horizontal_cat:bool,z_index_of_original:bool=0)->Board:
    """
    This function takes a Board and returns a new Board with the input board doubled.
    If is_horizontal_cat is True, the board is horizontally concatenated otherwise concatenation is vertical.
    Separation is the number of pixels between the two boards.
    This function supports negative separation values. In this case, one board is cropped to accomodate the other.
    if separation is negative, the z_index_of_original is used to determine which board lays ontop of the other.
    """

    np_board = boardIn.np
    ret_board_size = [len(np_board),len(np_board[0])]
    if is_horizontal_cat:
        ret_board_size[1] = ret_board_size[1]*2 + separation
    else:
        ret_board_size[0] = ret_board_size[0]*2 + separation
    ret_board = np.zeros(ret_board_size)

    if is_horizontal_cat:
        if z_index_of_original == 0:
            ret_board[:,:len(np_board[0])] = np_board
            ret_board[:,-len(np_board[0]):] = np_board
        else:
            ret_board[:,-len(np_board[0]):] = np_board
            ret_board[:,:len(np_board[0])] = np_board

    else:
        if z_index_of_original == 0:
            ret_board[:len(np_board),:] = np_board
            ret_board[-len(np_board):,:] = np_board
        else:
            ret_board[-len(np_board):,:] = np_board
            ret_board[:len(np_board),:] = np_board

    return Board(__root__ = ret_board.tolist())

def doubleInputBoard(board_pair:BoardPair,separation:int,is_horizontal_cat:bool,z_index_of_original:bool=0)->BoardPair:
    board_pair.input = doubleBoard(board_pair.input,separation,is_horizontal_cat,z_index_of_original)
    return board_pair

def rotate(board_pair:BoardPair, num_rotations:int=0) -> BoardPair:
    '''
    This function takes a Board and returns a new Board that is rotated 90 degrees `num_rotations` times
    '''
    assert num_rotations >= 0 and num_rotations < 4
    if num_rotations == 0:
        return board_pair
    return BoardPair(
        input=Board(__root__= np.rot90(board_pair.input.np, num_rotations).tolist()),
        output=Board(__root__= np.rot90(board_pair.output.np, num_rotations).tolist())
    )



def reflect(board_pair:BoardPair, x_axis:bool=False, y_axis:bool=False) -> BoardPair:
    '''
    This function takes a Board and returns a new Board that is reflected over one of the following: x-axis,y-axis, xy-axes, no axes
    '''

    if not x_axis and not y_axis:
        return board_pair.copy()

    if x_axis and y_axis:
        flip = None # None flips both axes
    elif x_axis:
        flip = 0
    elif y_axis:
        flip = 1

    return BoardPair(
        input=Board(__root__= np.flip(board_pair.input.np, flip).tolist()),
        output=Board(__root__= np.flip(board_pair.output.np, flip).tolist())
    )
def padInputOutput(board:BoardPair,size:int,pad_value:int)->BoardPair:
    input_np_board = board.input.np
    output_np_board = board.output.np
    input_np_board = padBoard(input_np_board,size,pad_value)
    output_np_board = padBoard(output_np_board,size,pad_value)
    return BoardPair(input=input_np_board.tolist(),output=output_np_board.tolist())

def padInputOnly(board:BoardPair,size:int,pad_value:int)->BoardPair:
    input_np_board = board.input.np
    input_np_board = padBoard(input_np_board,size,pad_value)
    return BoardPair(input=input_np_board.tolist(),output=board.output)

def padBoard(boardIn:np.ndarray,size:int,pad_value:int)->np.ndarray:
    boardIn = np.pad(boardIn,((size,size),(size,size)),constant_values = pad_value)
    return boardIn

def superResolution(bp:BoardPair,factor:int,stretch_axis:Direction)-> BoardPair:
    bp.input = superResolutionBoard(bp.input,factor,stretch_axis)
    bp.output = superResolutionBoard(bp.output,factor,stretch_axis)
    return bp

def superResolutionBoard(boardIn:Board,factor:int,stretch_axis:Direction)->Board:
    """
    This function takes a Board and returns a new Board that is super-resolved.
    The super-resolution is done by stretching the board in the specified direction.
    """
    np_board = boardIn.np
    if stretch_axis == Direction.horizontal:
        np_board = np.repeat(np_board,factor,axis=1)
    elif stretch_axis == Direction.vertical:
        np_board = np.repeat(np_board,factor,axis=0)
    elif stretch_axis == Direction.both:
        np_board = np.repeat(np_board,factor,axis=0)
        np_board = np.repeat(np_board,factor,axis=1)
    return Board(__root__ = np_board.tolist())

def taurusTranslate(boardPair:BoardPair, x_to_rem_choice, y_to_rem_choice, dir_choice_x, dir_choice_y)->BoardPair:
    input_board = taurusTranslateBoard(boardPair.input,x_to_rem_choice,y_to_rem_choice,dir_choice_x,dir_choice_y)
    output_board = taurusTranslateBoard(boardPair.output,x_to_rem_choice,y_to_rem_choice,dir_choice_x,dir_choice_y)
    return BoardPair(input=input_board,output=output_board)


def taurusTranslateBoard(boardIn:Board, x_translate, x_dir, y_translate, y_dir)->Board:
    """
    This function takes a Board and returns a new Board that is translated by the specified amount.
    The translation is done by shifting the board in the specified direction.
    """
    np_board = boardIn.np
    x_translate = x_translate * x_dir
    y_translate = y_translate * y_dir
    np_board = np.roll(np_board, x_translate, axis=1)
    np_board = np.roll(np_board, y_translate,axis=0)
    return Board(__root__ = np_board.tolist())

def quasiRotateBoard(boardIn:Board, angle)-> Board:
    x_component = np.cos(np.deg2rad(angle))*np.arange(len(boardIn.np[0]))
    y_component = np.sin(np.deg2rad(angle))*np.arange(len(boardIn.np))
    point_list = []
    translated_point_list = []
    for x in len(boardIn.np[0]):
        for y in len(boardIn.np):
            point_list.append(boardIn.np[x][y])
    for point in point_list:






