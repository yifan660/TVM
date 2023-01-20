template<typename T, typename=typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
class Array : public ObjectRef  {
    public:
        using value_type = T;
        Array() { data_ = ArrayNode::Empty(); }
        Array(Array<T>&& other) : ObjectRef()   {
            data_ = std::move(other.data_);
        }
        Array(const Array<T>& other) : ObjectRef()  {
            data_ = other.data_;
        }

        explicit Array(ObjectPtr<Object> n) : ObjectRef(n)  {}
}
