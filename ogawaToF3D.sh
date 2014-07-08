sed -i 's/<Alembic\/Util\/\(.*\)>/\"Util\1\"/g' include/*.h
sed -i 's/<Alembic\/Ogawa\/\(.*\)>/\"\1\"/g' include/*.h
sed -i 's/<Alembic\/Ogawa\/\(.*\)>/\"\1\"/g' src/*.cpp
