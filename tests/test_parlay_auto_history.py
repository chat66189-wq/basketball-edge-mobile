from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


class ParlayAutomaticHistoryTests(unittest.TestCase):
    def test_best_effort_is_removed_from_automatic_ui(self):
        self.assertNotIn('id="best-effort-mode"', HTML)
        self.assertIn("best_effort:false", HTML)

    def test_missing_history_uses_espn_not_sportapi_automatically(self):
        self.assertIn("autoTopUpHistoryForFixtures(data)", HTML)
        self.assertIn("/api/espn/history-sync?", HTML)
        self.assertIn("providerRequest('/api/espn/history-sync?", HTML)
        auto_function = HTML.split("async function autoTopUpHistoryForFixtures(analysis){", 1)[1].split(
            "const sportApiFixturesMessage=", 1
        )[0]
        self.assertNotIn("sportApiRequest(", auto_function)
        self.assertNotIn("/api/sportapi/", auto_function)
        self.assertIn("No se consultó SportAPI.ai", auto_function)

    def test_automatic_espn_merge_deduplicates_games_and_stats(self):
        self.assertIn("persistEspnHistoryWithoutDuplicates", HTML)
        self.assertIn("gameSignature", HTML)
        self.assertIn("existingStatKeys", HTML)
        self.assertIn("bea-v20-espn-history:", HTML)

    def test_automatic_history_sync_is_chunked_and_resumable(self):
        self.assertIn("bea-v20-espn-history-sync-job-v1", HTML)
        self.assertIn("CHUNK_DAYS=28", HTML)
        self.assertIn("max_games_per_league:String(job.maxGamesPerChunk)", HTML)
        self.assertIn("await persistEspnHistoryWithoutDuplicates(data)", HTML)
        self.assertIn("start_date:range.start", HTML)
        self.assertIn("end_date:range.end", HTML)
        self.assertIn("Reanudar sincronización", HTML)
        self.assertIn("completedChunks", HTML)

    def test_automatic_history_resume_uses_saved_chunk_progress(self):
        self.assertIn("runSavedHistorySyncJob", HTML)
        self.assertIn("localStorage.setItem(ESPN_SYNC_JOB_KEY", HTML)
        self.assertIn("localStorage.getItem(ESPN_SYNC_JOB_KEY)", HTML)

    def test_automatic_search_retries_after_new_team_boxscores(self):
        self.assertIn("if(updated.teamStatsRows>0)", HTML)
        self.assertIn("data=await requestRecommendation(games,teamStats)", HTML)
        self.assertIn("minimum_history_games:12", HTML)


if __name__ == "__main__":
    unittest.main()
