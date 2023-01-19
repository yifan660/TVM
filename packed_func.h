class TVMPODValue_  {
    public:
        operator double() const {
            if(type_code_ == kDLInt)    {
                return static_cast<double>(value_.v_int64);
            }
            TVM_CHECK_TYPE_CODE(type_code_, kDLFloat);
            return value_.v_float64;
        }

        operator int64_t() const    {
            TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
            return value_.v_int64;
        }

        operator uint64_t() const   {
            TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
            return value_.v_int64;
        }

        operator int() const    {
            TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
            ICHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
            ICHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
            return static_cast<int>(value_.v_int64);
        }

        operator bool() const   {
            TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
            return value_.v_int64 != 0;
        }

        operator void*() const  {
            if(type_code_ == kTVMNullptr)   return nullptr;
            if(type_code_ == kTVMDLTensorHandle)    return value_.v_handle;
            TVM_CHECK_TYPE_CODE(type_code_, kTVMOpaqueHandle);
            return value_.v_handle;
        }

        operator DLTensor*() const  {
            if(type_code_ == kTVMDLTensorHandle || type_code_ == kTVMNDArrayHandle) {
                return static_cast<DLTensor*>(value_.v_handle);
            } else  {
                if(type_code_ == kTVMNullptr) return nullptr;
                LOG(FATAL) << "Expected" << "DLTensor* or NDArray but got" << ArgTypeCode2Str(type_code_);
                return nullptr;
            }
        }

        operator NDArray() const    {
            if(type_code_ == kTVMNullptr)   return NDArray(ObjectPtr<Object>(nullptr));
            TVM_CHECK_TYPE_CODE(type_code_, kTVMNDArrayHandle);
            return NDArray(NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle)));
        }

        operator Module() const     {
            if(type_code_ == kTVMNullptr)   {
                return Module(ObjectPtr<Object>(nullptr));
            }
            TVM_CHECK_TYPE_CODE(type_code_, kTVMModuleHandle);
            return Module(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
        }

        operator PackedFunc() const {
            if(type_code_ == kTVMNullptr)   {
                return PackedFunc(ObjectPtr<Object>(nullptr));
            }
            TVM_CHECK_TYPE_CODE(type_code_, kTVMPackedFuncHandle);
            return PackedFunc(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
        }

        operator Device() const {
            TVM_CHECK_TYPE_CODE(type_code_, kDLDevice);
            return
        }
        
        int type_code() const   { return type_code_;}
        template<typename T>
        T* ptr() const  {
            return static_cast<T*>(value_.v_handle);
        }

        template<typename TObjectRef, typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
        inline bool IsObjectRef() const;
        template <typename TObjectRef>
        inline TObjectRef AsObjectRef() const;

    protected:
        friend class TVMArgsSetter;
        friend class TVMRetValue;
        friend class TVMMovableArgValue_;
        TVMPODValue_() : type_code_(kTVMNullptr)    {}
        TVMPODValue_(TVMValue value, int type_code) : value_(value), type_code_(type_code)  {}

        TVMValue value_;
        int type_code_;
};

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

        TVMRetValue& operator=(NDArray other)   {
            if(other.data_ != nullptr)  {
                this->clear();
                type_code_ = kTVMNDArrayHandle;
                value_.v_handle = NDArray::FFIGetHandle(other);
                ObjectRef::FFIClearAfterMove(&other);
            } else  {
                SwitchToPOD(kTVMNullptr);
                value_.v_handle = nullptr;
            }
            return *this;
        }

        TVMRetValue& operator=(Module m)    {
            SwitchToObject(kTVMModuleHandle, std::move(m.data_));
            return *this;
        }

        TVMRetValue& operator=(PackedFunc f)    {
            this->SwitchToObject(kTVMPackedFuncHandle, std::move(f.data_));
            return *this;
        }

        template<typename FType>
        TVMRetValue& operator=(const TypedPackedFunc<FType>& f)    {
            return operator=(f.packed());
        }

        TVMRetValue& operator=(const TVMRetValue& other)    {
            this->Assign(other);
            return *this;
        }

        TVMRetValue& operator=(const TVMArgValue& other)    {
            this->Assign(other);
            return *this;
        }    

        TVMRetValue& operator=(TVMMovableArgValue_&& other)    {
            this->Assign(other);
            return *this;
        }

        void MoveToCHost(TVMValue* ret_value, int* ret_type_code)  {
            ICHECK(type_code_ != kTVMStr && type_code_ != kTVMBytes);
            *ret_value = value_;
            *ret_type_code = type_code_;
            type_code_ = kTVMNullptr;
        }

        static TVMRetValue MoveFromCHost(TVMValue value, int type_code) {
            ICHECK(type_code <= kTVMPackedFuncHandle || type_code == kTVMNDArrayHandle);
            TVMRetValue ret;
            ret.value_ = value;
            ret.type_code_ = type_code;
            return ret;
        }

        const TVMValue& value() const   {
            ICHECK(type_code_ != kTVMObjectHandle && type_code_ != kTVMPackedFuncHandle && type_code_ != kTVMModuleHandle && type_code_ != kTVMStr) << "TVMRetValue.value can only be used for POD data";
            return value_;
        }

        // IsObjectRef will be called only when ObjectRef is base class of TObjectRef
        template<typename TObjectRef, typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
        inline TVMRetValue& operator=(TObjectRef other);
        template<typename T, typename = typename std::enable_if<std::is_class<T>::value>::type>
        inline operator T() const;

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
};
