class BufferFlattener : public StmtExprMutator  {
    public:
        static PrimFunc Flatten(PrimFunc func)  {
            Map<Var, Buffer> preflattened_buffer_map = Merge(func->buffer_map, func->preflattened_buffer_map);
            auto pass = BufferFlattener(func->buffer_map);
            auto writer = func.CopyOnWrite();
            writer->body = pass.VisitStmt();
            writer->preflattened_buffer_map = preflattened_buffer_map;
            writer->buffer_map = pass.updated_extern_buffer_map_;
            return func;
        }
    
    private:
        explicit BufferFlattener(const Map<Var, Buffer>& extern_buffer_map) {
            
        }
}
