import os
import sys
from src.logger.my_logging import logger


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    line_no=exc_tb.tb_lineno
    error_message=f"error occured in {file_name} &line number{line_no} with error{str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys) :
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    def __str__(self) -> str:
        return self.error_message

if __name__=="__main__":
    try:
        d= "b"/0
    except Exception as e:
        logger.error(f"error occured details are {e}")
        raise CustomException(e,sys)

 
