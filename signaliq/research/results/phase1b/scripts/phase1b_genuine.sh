#!/bin/bash
cd /home/user/as/signaliq
S=/tmp/claude-0/-home-user-as/be30be7d-2fbb-5fd2-a1f9-4ba855a85398/scratchpad; D=/home/user/as/signaliq/data
STORE=$D/phase1b_record/alpaca_responses.db
exec > $S/phase1b_genuine.out 2>&1
echo "HEAD $(git rev-parse --short HEAD); started $(date -u +%FT%TZ)"
rm -f $S/genuine_record_alpaca.log $S/genuine_replay_alpaca.log $S/genuine_record.txt $S/genuine_replay.txt $S/genuine_tables_*.txt $S/genuine_logical_*.txt $S/genuine_store_*.txt
[ -e $STORE ] && { echo "STORE ALREADY EXISTS: refusing (a genuine recording starts empty)"; echo DRIVER-DONE; exit 1; }
for r in record replay; do rm -f $D/phase1b_$r/universe_stage.db; .venv/bin/python scripts/seed_stage_database.py --parent $D/universe_stage.db --out $D/phase1b_$r/universe_stage.db > $S/seed2_$r.txt 2>&1 || { echo "seed $r FAILED"; echo DRIVER-DONE; exit 1; }; echo "seed $r: $(tail -1 $S/seed2_$r.txt)"; done
cmp -s $D/phase1b_record/universe_stage.db $D/phase1b_replay/universe_stage.db && echo "note: the two seeded files are byte-identical copies" || echo "note: the two seeded files differ at byte level"
ls -i $D/phase1b_record/universe_stage.db $D/phase1b_replay/universe_stage.db
echo "recording #3 (recordings #1 and #2 were interrupted; their stores are set aside, not used)"
echo "store present before start: $( [ -e $STORE ] && echo yes || echo no )"
echo "== GENUINE RECORD start $(date -u +%T) (empty store, live Alpaca requests)"
PYTHONPATH=. SIGNALIQ_SIP_RESEARCH_DATABASE_URL=sqlite:///$D/phase1b_record/universe_stage.db SIGNALIQ_SIP_RESEARCH_PARENT_DB=$D/universe_stage.db SIGNALIQ_SEC_OFFLINE=1 \
SIGNALIQ_ALPACA_RESPONSES=$STORE SIGNALIQ_ALPACA_RESPONSES_MODE=record \
.venv/bin/python -u $S/run_counted.py $S/genuine_record_alpaca.log --stage 100 --report $S/genuine_record.txt > $S/genuine_record.stdout 2>&1; echo "record exit $? $(date -u +%T)"
grep "STORE COUNTERS" $S/genuine_record_alpaca.log
echo "record log: request lines $(grep -c 'Alpaca request:' $S/genuine_record_alpaca.log); from-store lines $(grep -c 'Alpaca response from the store' $S/genuine_record_alpaca.log)"
.venv/bin/python scripts/response_store_tool.py verify $STORE > $S/genuine_store_verify.txt 2>&1; echo "verify exit $?"; cat $S/genuine_store_verify.txt
sha256sum $STORE > $S/genuine_store_sha.txt; cat $S/genuine_store_sha.txt
echo "== OFFLINE REPLAY start $(date -u +%T) (proxy -> 127.0.0.1:9, no credentials, NoNetworkTransport)"
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 NO_PROXY= no_proxy= \
PYTHONPATH=. SIGNALIQ_SIP_RESEARCH_DATABASE_URL=sqlite:///$D/phase1b_replay/universe_stage.db SIGNALIQ_SIP_RESEARCH_PARENT_DB=$D/universe_stage.db SIGNALIQ_SEC_OFFLINE=1 \
SIGNALIQ_ALPACA_RESPONSES=$STORE SIGNALIQ_ALPACA_RESPONSES_MODE=replay \
.venv/bin/python -u $S/run_counted.py $S/genuine_replay_alpaca.log --stage 100 --report $S/genuine_replay.txt > $S/genuine_replay.stdout 2>&1; echo "replay exit $? $(date -u +%T)"
grep "STORE COUNTERS" $S/genuine_replay_alpaca.log
echo "replay log: request lines $(grep -c 'Alpaca request:' $S/genuine_replay_alpaca.log); from-store lines $(grep -c 'Alpaca response from the store' $S/genuine_replay_alpaca.log)"
sha256sum -c $S/genuine_store_sha.txt && echo "STORE CHECKSUM UNCHANGED BY REPLAY"
cmp $S/genuine_record.txt $S/genuine_replay.txt && echo "REPORT BYTE-IDENTICAL (record vs replay)"
for r in record replay; do .venv/bin/python scripts/table_fingerprints.py $D/phase1b_$r/universe_stage.db --parent $D/universe_stage.db > $S/genuine_tables_$r.txt 2>&1; .venv/bin/python scripts/table_fingerprints.py $D/phase1b_$r/universe_stage.db --logical > $S/genuine_logical_$r.txt 2>&1; done
diff $S/genuine_tables_record.txt $S/genuine_tables_replay.txt && echo "ALL TABLES IDENTICAL (record vs replay)"
diff $S/genuine_logical_record.txt $S/genuine_logical_replay.txt && echo "ALL LOGICAL FINGERPRINTS IDENTICAL (record vs replay)"
echo "membership, decision by decision (record vs replay):"; .venv/bin/python $S/compare_membership.py $D/phase1b_record/universe_stage.db $D/phase1b_replay/universe_stage.db
echo "membership, decision by decision (record vs Stage 1.1):"; .venv/bin/python $S/compare_membership.py $D/phase1b_record/universe_stage.db $D/universe_stage.db
for r in record replay; do .venv/bin/python scripts/audit_completeness.py --db sqlite:///$D/phase1b_$r/universe_stage.db > $S/genuine_completeness_$r.txt 2>&1; echo "completeness audit $r exit $?: $(grep -E 'RESULT|PASS|FAIL' $S/genuine_completeness_$r.txt | tail -1)"; done
diff <(sed 1d $S/genuine_completeness_record.txt) <(sed 1d $S/genuine_completeness_replay.txt) && echo "COMPLETENESS AUDITS IDENTICAL (record vs replay, apart from the database name line)"
echo "completeness audit record vs Stage 1.1:"; diff $S/completeness_stage11.txt $S/genuine_completeness_record.txt && echo "identical to Stage 1.1 audit"
echo "record vs Stage 1.1 tables:"; diff $S/tables_B.txt $S/genuine_tables_record.txt
echo "record vs Stage 1.1 report:"; diff $S/stage11_runB.txt $S/genuine_record.txt
echo "record vs Phase 1 neutral report:"; cmp $S/phase1_neutral.txt $S/genuine_record.txt && echo "identical"
du -sh $D/phase1b_*/*; df -h / | tail -1
echo "finished $(date -u +%FT%TZ)"
echo DRIVER-DONE
