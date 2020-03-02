#include <ImageNetDatabase.h>
#include <kompex/KompexSQLiteDatabase.h>
#include <kompex/KompexSQLiteStatement.h>
#include <kompex/KompexSQLiteException.h>
#include <ConfigurationReader.h>
#include <boost/log/trivial.hpp>

using namespace neneta;
using namespace neneta::imh;

struct ImageNetDatabase::CashedData : public Kompex::SQLiteDatabase
{
    CashedData(const conf::ConfigurationReader& confReader)
        : Kompex::SQLiteDatabase(confReader.getStringParameter("configuration.persistance.trainsetdb"), SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
        , m_confReader(confReader)
        , m_sqlStatement(this)
        , m_selectOneBatch(this)
        , m_insertIntoImagesetTable(this)
        , m_selectImage(this)
        , m_insertIntoBatchTable(this)
        , m_deleteImage(this)
        , m_insertIntoProcessedTable(this)
        , m_clearBatchTable(this)
    {
    }

    ~CashedData()
    {
    }

    const conf::ConfigurationReader& m_confReader;
    Kompex::SQLiteStatement m_sqlStatement;
    Kompex::SQLiteStatement m_selectOneBatch;
    Kompex::SQLiteStatement m_insertIntoImagesetTable;
    Kompex::SQLiteStatement m_selectImage;
    Kompex::SQLiteStatement m_insertIntoBatchTable;
    Kompex::SQLiteStatement m_deleteImage;
    Kompex::SQLiteStatement m_insertIntoProcessedTable;
    Kompex::SQLiteStatement m_clearBatchTable;
};

ImageNetDatabase::ImageNetDatabase(const conf::ConfigurationReader &confReader)
    : m_database(std::make_shared<CashedData>(confReader))
{
    BOOST_LOG_TRIVIAL(debug) << "Initialising ImageNetDatabase.";
}

ImageNetDatabase::~ImageNetDatabase()
{
}

void ImageNetDatabase::startDb()
{
    try
    {
        if(m_database->m_confReader.getBooleanParameter("configuration.persistance.useexistingdb"))
        {
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE IF NOT EXISTS IMAGESET(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE IF NOT EXISTS BATCH(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE IF NOT EXISTS PROCESSED(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
        }
        else
        {
            m_database->m_sqlStatement.SqlStatement("DROP TABLE IF EXISTS IMAGESET");
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE IMAGESET(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
            m_database->m_sqlStatement.SqlStatement("DROP TABLE IF EXISTS BATCH");
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE IF NOT EXISTS BATCH(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
            m_database->m_sqlStatement.SqlStatement("DROP TABLE IF EXISTS PROCESSED");
            m_database->m_sqlStatement.SqlStatement("CREATE TABLE PROCESSED(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
        }
        m_database->MoveDatabaseToMemory();
        m_database->m_selectOneBatch.Sql("SELECT DISTINCT batchid FROM IMAGESET LIMIT 1");
        m_database->m_insertIntoImagesetTable.Sql("INSERT INTO IMAGESET VALUES(@batch, @img)");
        m_database->m_selectImage.Sql("SELECT image, rowid FROM IMAGESET WHERE batchid=@batch LIMIT 1");
        m_database->m_insertIntoBatchTable.Sql("INSERT INTO BATCH SELECT * FROM IMAGESET WHERE rowid = @row");
        m_database->m_deleteImage.Sql("DELETE FROM IMAGESET WHERE rowid = @row");
        m_database->m_insertIntoProcessedTable.Sql("INSERT INTO PROCESSED SELECT * FROM BATCH");
        m_database->m_clearBatchTable.Sql("DELETE FROM BATCH");
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in CashedData(): " << ex.GetString();
    }
}

void ImageNetDatabase::stopDb()
{
    m_database->m_selectOneBatch.FreeQuery();
    m_database->m_sqlStatement.FreeQuery();
    m_database->m_insertIntoImagesetTable.FreeQuery();
    m_database->m_selectImage.FreeQuery();
    m_database->m_insertIntoBatchTable.FreeQuery();
    m_database->m_deleteImage.FreeQuery();
    m_database->m_insertIntoProcessedTable.FreeQuery();;
    m_database->m_clearBatchTable.FreeQuery();
    m_database->SaveDatabaseFromMemoryToFile(m_database->m_confReader.getStringParameter("configuration.persistance.trainsetdb"));
}

IImageDatabase::BatchId ImageNetDatabase::getNextBatch() const
{
    IImageDatabase::BatchId rv = 0;
    try
    {
        if(m_database->m_selectOneBatch.FetchRow())
        {
            rv = m_database->m_selectOneBatch.GetColumnInt("batchid");
        }
        m_database->m_selectOneBatch.Reset();
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in getNextBatch(): " << ex.GetString();
    }

    return rv;
}

bool ImageNetDatabase::getNextImage(IImageDatabase::BatchId batch, IImageDatabase::Path& path) const
{
    bool rv = false;
    try
    {
        m_database->m_selectImage.BindInt(1, batch);
        if(m_database->m_selectImage.FetchRow())
        {
            path = m_database->m_selectImage.GetColumnString("image");
            int rowid = m_database->m_selectImage.GetColumnInt("rowid");

           // m_database->m_insertIntoBatchTable.fTransaction();
            //
            m_database->m_insertIntoBatchTable.BindInt(1, rowid);
            m_database->m_insertIntoBatchTable.Execute();
            m_database->m_insertIntoBatchTable.Reset();

            m_database->m_deleteImage.BindInt(1, rowid);
            m_database->m_deleteImage.Execute();
            m_database->m_deleteImage.Reset();
            //
          //  m_database->m_insertIntoBatchTable.CommitTransaction();
            rv = true;
        }
        m_database->m_selectImage.Reset();
    }
    catch(const Kompex::SQLiteException& ex)
    {
    //    m_database->m_insertIntoBatchTable.RollbackTransaction();
        BOOST_LOG_TRIVIAL(debug) << "Exception in getNextImage(): " << ex.GetString();
    }
    return rv;
}

void ImageNetDatabase::updateImage(const IImageDatabase::BatchId& batch, const IImageDatabase::Path& path)
{
    try
    {
        m_database->m_insertIntoImagesetTable.BindInt(1, batch);
        m_database->m_insertIntoImagesetTable.BindString(2, path);
        m_database->m_insertIntoImagesetTable.Execute();
        m_database->m_insertIntoImagesetTable.Reset();
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in updateImage(): " << ex.GetString();
    }
}

void ImageNetDatabase::restoreBatchTable()
{
    try
    {
        m_database->m_sqlStatement.SqlStatement("INSERT INTO IMAGESET SELECT * FROM BATCH");
        m_database->m_sqlStatement.SqlStatement("DELETE FROM BATCH");
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in restoreBatchTable(): " << ex.GetString();
    }
}

bool ImageNetDatabase::empty() const
{
    bool rv = false;
    try
    {
        m_database->m_sqlStatement.Sql("SELECT COUNT(*) AS cnt FROM PROCESSED");
        if(m_database->m_sqlStatement.FetchRow())
        {
            if(m_database->m_sqlStatement.GetColumnInt("cnt") == 0)
            {
                rv = true;
            }
        }
        m_database->m_sqlStatement.FreeQuery();
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in empty(): " << ex.GetString();
    }
    return rv;
}


void ImageNetDatabase::restore()
{
    try
    {
        m_database->m_sqlStatement.SqlStatement("DELETE FROM BATCH");
        m_database->m_sqlStatement.SqlStatement("DROP TABLE IF EXISTS IMAGESET");
        m_database->m_sqlStatement.SqlStatement("ALTER TABLE PROCESSED RENAME TO IMAGESET");
        m_database->m_sqlStatement.SqlStatement("CREATE TABLE PROCESSED(batchid INTEGER NOT NULL, image TEXT NOT NULL PRIMARY KEY)");
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in restore(): " << ex.GetString();
    }
}


void ImageNetDatabase::setupPersistancePoint()
{
    try
    {
        //Return to IMAGESET from the BATCH if BATCH is not complete. Sync point is always at the end of the batch
        m_database->m_insertIntoProcessedTable.Execute();
        m_database->m_insertIntoProcessedTable.Reset();
        m_database->m_clearBatchTable.Execute();
        m_database->m_clearBatchTable.Reset();
    }
    catch(const Kompex::SQLiteException& ex)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in setupPersistancePoint(): " << ex.GetString();
    }
}
