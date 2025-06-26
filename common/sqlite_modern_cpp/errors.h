#pragma once

#include <string>
#include <stdexcept>

#include <sqlite3.h>

namespace sqlite {

	class sqlite_exception: public std::runtime_error {
	public:
		sqlite_exception(const char* msg, str_ref sql, int code = -1): runtime_error(msg), code(code), sql(sql) {}
		sqlite_exception(int code, str_ref sql, const char *msg = nullptr): runtime_error(msg ? msg : sqlite3_errstr(code)), code(code), sql(sql) {}
		int get_code() const {return code & 0xFF;}
		int get_extended_code() const {return code;}
		std::string get_sql() const {return sql;}
		const char *errstr() const {return code == -1 ? "Unknown error" : sqlite3_errstr(code);}
	private:
		int code;
		std::string sql;
	};

	namespace errors {
		//One more or less trivial derived error class for each SQLITE error.
		//Note the following are not errors so have no classes:
		//SQLITE_OK, SQLITE_NOTICE, SQLITE_WARNING, SQLITE_ROW, SQLITE_DONE
		//
		//Note these names are exact matches to the names of the SQLITE error codes.
#define SQLITE_MODERN_CPP_ERROR_CODE(NAME,name,derived) \
		class name: public sqlite_exception { using sqlite_exception::sqlite_exception; };\
		derived
#define SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED(BASE,SUB,base,sub) \
		class base ## _ ## sub: public base { using base::base; };
#include "lists/error_codes.h"
#undef SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED
#undef SQLITE_MODERN_CPP_ERROR_CODE

		//Some additional errors are here for the C++ interface
		class more_rows: public sqlite_exception { using sqlite_exception::sqlite_exception; };
		class no_rows: public sqlite_exception { using sqlite_exception::sqlite_exception; };
		class more_statements: public sqlite_exception { using sqlite_exception::sqlite_exception; }; // Prepared statements can only contain one statement
		class invalid_utf16: public sqlite_exception { using sqlite_exception::sqlite_exception; };
		class unknown_binding: public sqlite_exception { using sqlite_exception::sqlite_exception; };

		static void throw_sqlite_error(const int& error_code, str_ref sql = "", const char *errmsg = nullptr) {
			switch(error_code & 0xFF) {
#define SQLITE_MODERN_CPP_ERROR_CODE(NAME,name,derived)     \
				case SQLITE_ ## NAME: switch(error_code) {          \
					derived                                           \
					case SQLITE_ ## NAME:                             \
					default: throw name(error_code, sql);             \
				}

#if SQLITE_VERSION_NUMBER < 3010000
#define SQLITE_IOERR_VNODE (SQLITE_IOERR | (27<<8))
#define SQLITE_IOERR_AUTH (SQLITE_IOERR | (28<<8))
#define SQLITE_AUTH_USER (SQLITE_AUTH | (1<<8))
#endif

#define SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED(BASE,SUB,base,sub) \
					case SQLITE_ ## BASE ## _ ## SUB: throw base ## _ ## sub(error_code, sql, errmsg);
#include "lists/error_codes.h"
#undef SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED
#undef SQLITE_MODERN_CPP_ERROR_CODE
				default: throw sqlite_exception(error_code, sql, errmsg);
			}
		}
	}
	namespace exceptions = errors;
}
