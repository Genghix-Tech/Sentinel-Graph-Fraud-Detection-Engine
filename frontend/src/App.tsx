import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Alerts from "./pages/Alerts";
import Investigation from "./pages/Investigation";
import Models from "./pages/Models";
import Overview from "./pages/Overview";

export default function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/favicon.svg" width={30} height={30} alt="" />
          <div>
            <b>SentinelGraph</b>
            <small>Fraud investigation engine</small>
          </div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/alerts">Alert queue</NavLink>
          <NavLink to="/models">Models</NavLink>
        </nav>
        <div className="foot">
          FastAPI · XGBoost · PyG · NetworkX · SHAP
          <br />
          <a href="/docs" target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>API docs ↗</a>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/alerts/:id" element={<Investigation />} />
          <Route path="/models" element={<Models />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}
