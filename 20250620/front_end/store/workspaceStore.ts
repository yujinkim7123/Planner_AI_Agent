// store/workspaceStore.ts

import { create } from 'zustand';

// 1. workspace에 포함될 데이터의 타입(모양)을 정의합니다.
// 백엔드 main.py의 create_new_workspace와 동기화
interface WorkspaceState {
  sessionId: string | null;
  history: any[];
  artifacts: {
    product_type: string | null;
    retrieved_data: any; // data_retriever 결과 전체
    product_data: any;   // fetch_product_context 결과
    sensor_data: any;    // fetch_sensor_context 결과
    columns_product: any; // get_columns_for_product 결과
    cx_ward_clustering_results: any;
    cx_sna_results: any[];
    cx_lda_results: any; // 이제 단일 객체 (내부에 list 포함)
    cx_opportunity_scores: any[];
    cx_cam_results: any[];
    personas: any[];
    selected_persona: any;
    service_ideas: any; // { service_ideas: [...] } 형태
    selected_service_idea: any;
    data_plan_for_service: any[]; // 배열로 변경
    selected_data_plan_for_service: any;
    cdp_definition: any[]; // 배열로 변경
    selected_cdp_definition: any;
    data_plan_recommendation_message: any;
    // conversation_state는 프롬프트 주입에만 사용되므로 클라이언트 상태 불필요
  };
}

// 2. 이 데이터를 변경할 수 있는 '행동(Action)'의 타입을 정의합니다.
interface WorkspaceActions {
  setWorkspace: (workspace: Partial<WorkspaceState>) => void;
  setSessionId: (sessionId: string) => void;
  addMessageToHistory: (message: any) => void;
  resetWorkspace: () => void;
}

// 3. 초기 상태값을 정의합니다. 앱이 처음 시작될 때의 기본값입니다.
const initialState: WorkspaceState = {
  sessionId: null,
  history: [],
  artifacts: {
    product_type: null,
    retrieved_data: null,
    product_data: null,
    sensor_data: null,
    columns_product: null,
    cx_ward_clustering_results: null,
    cx_sna_results: [],
    cx_lda_results: null,
    cx_opportunity_scores: [],
    cx_cam_results: [],
    personas: [],
    selected_persona: null,
    service_ideas: null,
    selected_service_idea: null,
    data_plan_for_service: [], // 배열로 초기화
    selected_data_plan_for_service: null,
    cdp_definition: [], // 배열로 초기화
    selected_cdp_definition: null,
    data_plan_recommendation_message: null,
  },
};

// 4. Zustand 스토어를 생성합니다.
export const useWorkspaceStore = create<WorkspaceState & WorkspaceActions>((set) => ({
  ...initialState,

  // 서버에서 받은 workspace 전체 또는 일부로 상태를 업데이트하는 액션
  setWorkspace: (workspace) => set((state) => ({ ...state, ...workspace })),
  
  // 세션 ID를 설정하는 액션
  setSessionId: (sessionId) => set({ sessionId }),

  // 대화 기록에 메시지를 추가하는 액션
  addMessageToHistory: (message) => 
    set((state) => ({ history: [...state.history, message] })),

  // 모든 상태를 초기값으로 되돌리는 리셋 액션
  resetWorkspace: () => set(initialState),
}));