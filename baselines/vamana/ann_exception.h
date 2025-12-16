// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <string>
#include <sstream>

#ifndef _WINDOWS
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace diskann {
  class ANNException {
   public:
    ANNException(const std::string& message, int errorCode) : _errorCode(errorCode), _message(message), _funcSig(""), _fileName(""),
    _lineNum(0) {
}
    ANNException(const std::string& message, int errorCode,
                                   const std::string& funcSig,
                                   const std::string& fileName,
                                   unsigned int       lineNum) : ANNException(message, errorCode) {
                                    _funcSig = funcSig;
                                    _fileName = fileName;
                                    _lineNum = lineNum;
                                  }

    std::string message() const {
      std::stringstream sstream;
  
      sstream << "Exception: " << _message;
      if (_funcSig != "")
        sstream << ". occurred at: " << _funcSig;
      if (_fileName != "" && _lineNum != 0)
        sstream << " defined in file: " << _fileName << " at line: " << _lineNum;
      if (_errorCode != -1)
        sstream << ". OS error code: " << std::hex << _errorCode;
  
      return sstream.str();
    }

   private:
    int          _errorCode;
    std::string  _message;
    std::string  _funcSig;
    std::string  _fileName;
    unsigned int _lineNum;
  };
}  // namespace diskann
