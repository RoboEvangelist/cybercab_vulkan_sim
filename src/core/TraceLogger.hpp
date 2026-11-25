#pragma once

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <fstream>
#include <string>

namespace cybercab {

class TraceLogger {
public:
    TraceLogger() = default;
    ~TraceLogger();

    bool open(const std::string& jsonlPath);
    void close();
    void logFrame(int stepIdx, float simTimeSec,
                  const SimConfig& cfg,
                  const EgoState& ego,
                  const PlannerCommand& cmd,
                  const PerceptionFrameOutput& perc,
                  const ContactEvent& contact);

    static std::string formatFrameJson(int stepIdx, float simTimeSec,
                                       const SimConfig& cfg,
                                       const EgoState& ego,
                                       const PlannerCommand& cmd,
                                       const PerceptionFrameOutput& perc,
                                       const ContactEvent& contact);

    bool isOpen() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

} // namespace cybercab
