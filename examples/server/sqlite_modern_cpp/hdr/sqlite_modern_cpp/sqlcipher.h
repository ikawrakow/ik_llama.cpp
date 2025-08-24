#pragma once

#ifndef SQLITE_HAS_CODEC
#define SQLITE_HAS_CODEC
#endif

#include "../sqlite_modern_cpp.h"

namespace sqlite {
	struct sqlcipher_config : public sqlite_config {
		std::string key;
	};

	class sqlcipher_database : public database {
	public:
		sqlcipher_database(std::string db, const sqlcipher_config &config): database(db, config) {
			set_key(config.key);
		}
		
		sqlcipher_database(std::u16string db, const sqlcipher_config &config): database(db, config) {
			set_key(config.key);
		}
		
		void set_key(const std::string &key) {
			if(auto ret = sqlite3_key(_db.get(), key.data(), key.size()))
				errors::throw_sqlite_error(ret);
		}

		void set_key(const std::string &key, const std::string &db_name) {
			if(auto ret = sqlite3_key_v2(_db.get(), db_name.c_str(), key.data(), key.size()))
				errors::throw_sqlite_error(ret);
		}

		void rekey(const std::string &new_key) {
			if(auto ret = sqlite3_rekey(_db.get(), new_key.data(), new_key.size()))
				errors::throw_sqlite_error(ret);
		}

		void rekey(const std::string &new_key, const std::string &db_name) {
			if(auto ret = sqlite3_rekey_v2(_db.get(), db_name.c_str(), new_key.data(), new_key.size()))
				errors::throw_sqlite_error(ret);
		}
	};
}
