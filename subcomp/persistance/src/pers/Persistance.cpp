#include <Persistance.h>
#include <kompex/KompexSQLiteDatabase.h>
#include <kompex/KompexSQLiteStatement.h>
#include <kompex/KompexSQLiteException.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>

using namespace neneta;
using namespace neneta::pers;

struct Persistance::PersistanceImpl : public Kompex::SQLiteDatabase
{
public:
    PersistanceImpl(const std::string& dbPath)
        : Kompex::SQLiteDatabase(dbPath, SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
        , m_sqlStatement(this)
        , m_storeBlob(this)
        , m_restoreBlob(this)
        , m_deleteBlob(this)
    {
        try
        {
            m_sqlStatement.SqlStatement("CREATE TABLE IF NOT EXISTS LAYERS(id TEXT NOT NULL PRIMARY KEY, size INTEGER NOT NULL, data BLOB)");
            m_storeBlob.Sql("INSERT INTO LAYERS VALUES(@id, @size, @blob)");
            m_restoreBlob.Sql("SELECT size, data FROM LAYERS WHERE ID = @id");
            m_deleteBlob.Sql("DELETE FROM LAYERS WHERE id = @id");
        }
        catch(const Kompex::SQLiteException& ex)
        {
            BOOST_LOG_TRIVIAL(debug) << "Exception in PersistanceImpl(): " << ex.GetString();
        }
    }

    ~PersistanceImpl()
    {
        m_sqlStatement.FreeQuery();
        m_storeBlob.FreeQuery();
        m_restoreBlob.FreeQuery();
        m_deleteBlob.FreeQuery();
    }

    void storeFloatBlob(const std::string& id, const std::vector<cmn::GPUFLOAT>& blob)
    {
        try
        {
            m_storeBlob.BindString(1, id);
            m_storeBlob.BindInt(2, blob.size());
            m_storeBlob.BindBlob(3, blob.data(), blob.size()*sizeof(cmn::GPUFLOAT));
            m_storeBlob.Execute();
            m_storeBlob.Reset();
        }
        catch(const Kompex::SQLiteException& ex)
        {
            BOOST_LOG_TRIVIAL(debug) << "Exception in storeFloatBlob(): " << ex.GetString();
        }
    }

    void restoreFloatBlob(const std::string& id, std::vector<cmn::GPUFLOAT>& blob) const
    {
        try
        {
            m_restoreBlob.BindString(1, id);
            if(m_restoreBlob.FetchRow())
            {
                size_t size = m_restoreBlob.GetColumnInt("size");
                const cmn::GPUFLOAT* pdata = static_cast<const cmn::GPUFLOAT*>(m_restoreBlob.GetColumnBlob("data"));
                blob.assign(pdata, pdata + size);
            }
            m_restoreBlob.Reset();
        }
        catch(const Kompex::SQLiteException& ex)
        {
            BOOST_LOG_TRIVIAL(debug) << "Exception in restoreFloatBlob(): " << ex.GetString();
        }

    }

    void removeFloatBlob(const std::string& id)
    {
        try
        {
            m_deleteBlob.BindString(1, id);
            m_deleteBlob.Execute();
            m_deleteBlob.Reset();
        }
        catch(const Kompex::SQLiteException& ex)
        {
            BOOST_LOG_TRIVIAL(debug) << "Exception in removeFloatBlob(): " << ex.GetString();
        }
    }

private:
    Kompex::SQLiteStatement m_sqlStatement;
    Kompex::SQLiteStatement m_storeBlob;
    Kompex::SQLiteStatement m_restoreBlob;
    Kompex::SQLiteStatement m_deleteBlob;
};

Persistance::Persistance(const conf::ConfigurationReader& confReader)
    : m_impl(std::make_shared<PersistanceImpl>(confReader.getStringParameter("configuration.persistance.netconfdb")))
    , m_storeActive(confReader.getBooleanParameter("configuration.persistance.store"))
    , m_restoreActive(confReader.getBooleanParameter("configuration.persistance.restore"))

{
}

void Persistance::storeFloatBlob(const std::string& id, const std::vector<cmn::GPUFLOAT>& blob)
{
    if(m_storeActive)
    {
        BOOST_LOG_TRIVIAL(debug) << "storeFloatBlob() for id " << id << " of size " << blob.size() << " cmn::GPUFLOATs.";
        m_impl->removeFloatBlob(id);
        m_impl->storeFloatBlob(id, blob);
    }
}

void Persistance::restoreFloatBlob(const std::string& id, std::vector<cmn::GPUFLOAT>& blob) const
{
    if(m_restoreActive)
    {
        m_impl->restoreFloatBlob(id, blob);
        BOOST_LOG_TRIVIAL(debug) << "restoreFloatBlob() for id " << id << " of size " << blob.size();
    }
}


