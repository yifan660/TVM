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
}
