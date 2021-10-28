import React from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import UploadModal from "./components/UploadModal";
import ProjectModal from "./components/ProjectModal";
import LoadProjectModal from "./components/LoadProjectModal";
import GHPagesRedirect from "./components/GHPagesRedirect";
import ShowApplyMappingsModal from "./components/ApplyMappingsModal";
import CodeExportModal from "./components/CodeExportModal";
import DatasetPage from "./pages/DatasetPage";
import ColumnPage from "./pages/ColumnPage";
import WelcomePage from "./pages/WelcomePage";
import AboutPage from "./pages/AboutPage";
import ProjectPage from "./pages/ProjectPage";
import "typeface-lato";
import { BrowserRouter as Router, Route } from "react-router-dom";
import { TermsOfService, Privacy } from "@dataclinic/terms";

import "./App.scss";

function App() {
  return (
    <Router basename={process.env.PUBLIC_URL}>
      <GHPagesRedirect />
      <div className="App">
        <div className="main">
          <Route
            path="/project/:projectID/dataset/:datasetID"
            exact={true}
            component={DatasetPage}
          />
          <Route
            path="/project/:projectID/column/:columnID"
            exact={true}
            component={ColumnPage}
          />
          <Route
            path="/project/:projectID"
            exact={true}
            component={ProjectPage}
          />
          <Route path="/about" exact={true} component={AboutPage} />
          <Route path="/" exact={true} component={WelcomePage} />
          <Route path="/privacy">
            <div style={{ textAlign: "left" }}>
              <Privacy />
            </div>
          </Route>
          <Route path="/terms">
            <div style={{ textAlign: "left" }}>
              <TermsOfService />
            </div>
          </Route>
        </div>
        <Footer />

        <Route path="/" component={Header} />

        <Route path="/new_project" component={ProjectModal} />
        <Route path="/load_project" component={LoadProjectModal} />

        <Route path="/project/:projectID/export" component={CodeExportModal} />
        <Route
          path="/project/:projectID/add_datasets"
          component={UploadModal}
        />

        <Route
          path="/dataset/:datasetID/apply"
          component={ShowApplyMappingsModal}
        />
      </div>
    </Router>
  );
}

export default App;
