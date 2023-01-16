class TVMRetValue : public TVMPODValue_ {
    public:
        TVMRetValue()   {}
        TVMRetValue(TVMRetValue&& other) : TVMPODValue_(other.value_, other.type_code_) {
            other.value_.v_handle = nullptr;
            other.type_code_ = kTVMNullptr;
        }
        ~TVMRetValue()

        void SwitchToPOD()  {
            if(type_code_!=type_code)   {
                this->Clear();
                type_code_ = type_code;
            }
        }
        template<typename T>
        void SwitchToClass(int type_code, T v)  {
            if(type_code_!=type_code)   {
                this->Clear();
                type_code_ = type_code;
                value_.v_handle = new T(v);
            }
        }

        using TVMPODValue_::operator double;
        using TVMPODValue_::operator int64_t;
        using TVMPODValue_::operator uint64_t;
        using TVMPODValue_::operator int;
        using TVMPODValue_::operator bool;
        using TVMPODValue_::operator void*;
        using TVMPODValue_::operator DLTensor*;
        using TVMPODValue_::operator Device;
        using TVMPODValue_::operator NDArray;
        using TVMPODValue_::operator Module;
        using TVMPODValue_::operator PackedFunc;
        
}

