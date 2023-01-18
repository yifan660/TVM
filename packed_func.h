class TVMRetValue : public TVMPODValue_ {
    public:
        TVMRetValue()   {}
        TVMRetValue(TVMRetValue&& other) : TVMPODValue_(other.value_, other.type_code_) {
            other.value_.v_handle = nullptr;
            other.type_code_ = kTVMNullptr;
        }
        ~TVMRetValue()  {this->Clear();}
        
        using TVMPODValue_::operator double;
        using TVMPODValue_::operator int64_t;
        using TVMPODValue_::operator uint6_t;
        using TVMPODValue_::operator int;
        using TVMPODValue_::operator bool;
        using TVMPODValue_::operator void*;
        using TVMPODValue_::operator DLTensor*;
        using TVMPODValue_::operator Device;
        using TVMPODValue_::operator NDArray;
        using TVMPODValue_::operator Module;
        using TVMPODValue_::operator PackedFunc;
        using TVMPODValue_::AsObjectRef;
        using TVMPODValue_::IsObjectRef;

        TVMRetvalue(const TVMRetValue& other) : TVMPODValue_()  {this->Assign(other);}
        operator std::string() const    {
            if(type_code_ == kTVMDataType)  {
                return DLDataType2String(operator DLDataType());
            } else if(type_code_ == kTVMBytes)  {
                return *ptr<std::string>();
            }
            TVM_CHECK_TYPE_CODE(type_code_, kTVMStr);
            return *ptr<std::string>();
        }

        operator DLDataType() const {
            if(type_code_ == kTVMStr)   {
                return String2DLDataType(operator std::string());
            }
            TVM_CHECK_TYPE_CODE(type_code_, kTVMDataType);
            return value_.v_type;
        }

        operator DataType() const   {return DataType(operator DLDataType());}
        template <typename FType>
        operator TypedPackedFunc<FType>() const {
            return TypedPackedFunc<FType>(operator PackedFunc());
        }

        TVMRetValue& operator=(TVMRetValue&& other) {
            this->Clear();
            value_ = other.value_;
            type_code_ = other.type_code_;
            other.type_code_ = kTVMNullptr;
            return *this;
        } 
        TVMRetValue& operator=(double value)    {
            this->SwitchToPOD(kDLFloat);
            value_.v_float64 = value;
            return *this;
        }
        TVMRetValue& operator=(std::nullptr_t value)    {
            this->SwitchToPOD(kTVMNullptr);
            value_.v_handle = value;
            return *this;
        }
        TVMRetValue& operator=(void* value) {
            this->SwitchToPOD(kTVMOpaqueHandle);
            value_.v_handle = value;
            return *this;
        }
        TVMRetValue& operator=(int64_t value) {
            this->SwitchToPOD(kDLInt);
            value_.v_int64 = value;
            return *this;
        }
        TVMRetValue& operator=(int value) {
            this->SwitchToPOD(kDLInt);
            value_.v_int64 = value;
            return *this;
        }
        TVMRetValue& operator=(DLDevice value) {
            this->SwitchToPOD(kDLDevice);
            value_.v_device = value;
            return *this;
        }
        TVMRetValue& operator=(DLDataType t)    {
            this->SwitchToPOD(kTVMDataType);
            value_.v_device = t;
            return *this;
        }

        TVMRetValue& operator=(bool value)    {
            this->SwitchToPOD(kDLInt);
            value_.v_int64 = value;
            return *this;
        }

        TVMRetValue& operator=(std::string value)    {
            this->SwitchToClass(kTVMStr, value);
            return *this;
        }

        TVMRetValue& operator=(TVMByteArray value)    {
            this->SwitchToClass(kTVMBytes, std::string(value.data, value.size));
            return *this;
        }

        // IsObjectRef will be called only when ObjectRef is base class of TObjectRef
        template<typename TObjectRef, typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
        inline bool IsObjectRef() const;

    private:
        void SwitchToPOD(int type_code)  {
            if(type_code_!=type_code)   {
                this->Clear();
                type_code_ = type_code;
            }
        }

        template<typename T>
        void SwitchToClass(int type_code, T v)    {
            if(type_code_!=type_code)    {
                this->Clear();
                type_code_ = type_code;
                value_.v_handle = new T(v);
            } else  {
                *static_cast<T*>(value_.v_handle) = v;
            }
        }

        void SwitchToObject(int type_code, ObjectPtr<Object> other)    {
            if(other.data_!=nullptr)    {
                this->Clear();
                type_code_ = type_code;
                value_.v_handle = other.data_;
                other.data_ = nullptr;
            } else  {
                SwitchToPOD(kTVMNullptr);
                value_.v_handle = nullptr;
            }
        }

        template<typename T>
        void Assign(const T& other)   {
            switch(other.type_code_)    {
                case kTVMStr:
                    SwitchToClass<std::string>(kTVMStr, other);
                    break;
                case kTVMBytes:
                    SwitchToClass<std::string>(kTVMBytes, other);
                    break;
                case kTVMPackedFuncHandle:
                    *this = other.operator PackedFunc();
                    break;
                case kTVMModuleHandle:
                    *this = other.operator Module();
                    break;
                case kTVMNDArrayHandle:
                    *this = other.operator NDArray();
                    break;
                case kTVMObjectHandle:
                    SwitchToObject(kTVMObjectHandle, GetObjectPtr<Object>(static_cast<Object*>(other.value_.v_handle)));
                    break;
                case kTVMObjectRValueRefArg: 
                    operator=(other.operator ObjectRef());
                    break;
                default:
                    SwitchToPOD(other.type_code());
                    value_ = other.value_;
                    break;
            
            }
        }
        void Clear()    {
            if(type_code_==kTVMNullptr)    return;
            switch(type_code_)    {
                case kTVMStr:
                case kTVMBytes:
                    delete ptr<std::string>();
                    break;
                case kTVMPackedFuncHandle:
                    static_cast<Object*>(value_.v_handle)->DecRef();
                    break;
                case kTVMNDArrayHandle:
                    NDArray::FFIDecRef(static_cast<TVMArrayHandle>(value_.v_handle));
                    break;
                case kTVMModuleHandle:
                    static_cast<Object*>*(value_.v_handle)->DecRef();
                    break;
                case kTVMObjectHandle:
                    static_cast<Object*>*(value_.v_handle)->DecRef();
                    break;                
            }
        }
}
