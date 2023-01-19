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
            for(const auto& kv : extern_buffer_map)   {
                updated_extern_buffer_map_.Set(kv.first, GetFlattenedBuffer(kv.second));
            }
        }

        Stmt VisitStmt_(const BlockNode* op) final  {
            ICHECK_EQ(op->match_buffer.size(),0) << "Unexpected MatchBufferRegion found during tir.transform.FlattenBuffer." << "All MatchBufferRegion should be removed in tir.transform.LowerMatchBuffer.";
            Block block = GetRef<Block>(op);
            Array<BufferRegion> reads = op->reads;
            reads.MutateByApply()
        }
}

class Allocate : public Stmt    {
    public:
        TVM_DLL Allocate(Var )
}
