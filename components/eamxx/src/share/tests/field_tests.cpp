#include <catch2/catch.hpp>
#include <numeric>

#include "ekat/kokkos/ekat_subview_utils.hpp"
#include "share/field/field_identifier.hpp"
#include "share/field/field_header.hpp"
#include "share/field/field.hpp"
#include "share/field/field_manager.hpp"
#include "share/field/field_utils.hpp"
#include "share/util/scream_setup_random_test.hpp"

#include "share/grid/point_grid.hpp"

#include "ekat/ekat_pack.hpp"
#include "ekat/ekat_pack_utils.hpp"
#include "ekat/util/ekat_test_utils.hpp"

namespace {

TEST_CASE("field_layout", "") {
  using namespace scream;
  using namespace ShortFieldTagsNames;

  using TVec = std::vector<FieldTag>;
  using IVec = std::vector<int>;

  FieldLayout fl1 ({COL},{1});
  FieldLayout fl2 ({COL,CMP},{1,1});
  FieldLayout fl3 ({COL,CMP,CMP},{1,3,4});
  FieldLayout fl4 ({COL,LEV},{1,1});
  FieldLayout fl5 ({COL,CMP,LEV},{1,1,1});
  FieldLayout fl6 ({COL,CMP,CMP,ILEV},{1,5,6,1});

  REQUIRE (fl1.type()==LayoutType::Scalar2D);
  REQUIRE (fl2.type()==LayoutType::Vector2D);
  REQUIRE (fl3.type()==LayoutType::Tensor2D);
  REQUIRE (fl4.type()==LayoutType::Scalar3D);
  REQUIRE (fl5.type()==LayoutType::Vector3D);
  REQUIRE (fl6.type()==LayoutType::Tensor3D);

  REQUIRE (not fl1.is_vector_layout());
  REQUIRE (    fl2.is_vector_layout());
  REQUIRE (not fl3.is_vector_layout());
  REQUIRE (not fl4.is_vector_layout());
  REQUIRE (    fl5.is_vector_layout());
  REQUIRE (not fl6.is_vector_layout());

  REQUIRE (not fl1.is_tensor_layout());
  REQUIRE (not fl2.is_tensor_layout());
  REQUIRE (    fl3.is_tensor_layout());
  REQUIRE (not fl4.is_tensor_layout());
  REQUIRE (not fl5.is_tensor_layout());
  REQUIRE (    fl6.is_tensor_layout());

  REQUIRE (fl2.get_vector_tag()==CMP);
  REQUIRE (fl5.get_vector_tag()==CMP);
  REQUIRE (fl2.get_vector_component_idx()==1);
  REQUIRE (fl5.get_vector_component_idx()==1);
  REQUIRE (fl2.get_vector_dim()==1);
  REQUIRE (fl5.get_vector_dim()==1);

  REQUIRE (fl3.get_tensor_tags()==TVec{CMP,CMP});
  REQUIRE (fl6.get_tensor_components_ids()==IVec{1,2});
  REQUIRE (fl3.get_tensor_dims()==IVec{3,4});
  REQUIRE (fl6.get_tensor_dims()==IVec{5,6});
}

TEST_CASE("field_identifier", "") {
  using namespace scream;
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  std::vector<FieldTag> tags1 = {EL, LEV, CMP};
  std::vector<FieldTag> tags2 = {EL, CMP, LEV};

  std::vector<int> dims1 = {2, 3, 4};
  std::vector<int> dims2 = {2, 4, 3};

  FieldIdentifier fid1 ("field_1", {tags1,dims1}, kg, "some_grid");
  FieldIdentifier fid2 ("field_1", {tags1,dims1}, kg, "some_grid");
  FieldIdentifier fid3 ("field_1", {tags1,dims2}, kg, "some_grid");
  FieldIdentifier fid4 ("field_2", {tags1,dims2}, kg, "some_grid");
  FieldIdentifier fid5 ("field_2", {tags2,dims2}, kg, "some_grid");
  FieldIdentifier fid6 ("field_2", {tags2,dims2}, m, "some_grid");
  FieldIdentifier fid7 ("field_2", {tags2,dims2}, m, "some_other_grid");

  REQUIRE (fid1==fid2);
  REQUIRE (fid2!=fid3);
  REQUIRE (fid3!=fid4);
  REQUIRE (fid4!=fid5);
  REQUIRE (fid5!=fid6);
  REQUIRE (fid6!=fid7);

  // Check that has_tag option works
  REQUIRE(fid1.get_layout().has_tag(CMP));
  REQUIRE(!fid1.get_layout().has_tag(GP));
}

TEST_CASE("field_tracking", "") {
  using namespace scream;

  FieldTracking track;
  util::TimeStamp time1(2021,10,12,17,8,10);
  util::TimeStamp time2(2021,10,12,17,8,20);
  track.update_time_stamp(time2);

  // Cannot rewind time (yet)
  REQUIRE_THROWS  (track.update_time_stamp(time1));
}

TEST_CASE("field", "") {
  using namespace scream;
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  using P4 = ekat::Pack<Real,4>;
  using P8 = ekat::Pack<Real,8>;
  using P16 = ekat::Pack<Real,16>;

  auto engine = setup_random_test ();
  using RPDF = std::uniform_real_distribution<Real>;
  RPDF pdf(0.01,0.99);

  std::vector<FieldTag> tags = {COL,LEV};
  std::vector<int> dims = {3,24};

  FieldIdentifier fid ("field_1", {tags,dims}, m/s,"some_grid");

  // Check if we can extract a reshaped view
  SECTION ("reshape") {
    Field f1 (fid);

    // Should not be able to reshape before allocating
    REQUIRE_THROWS(f1.get_view<Real*>());

    f1.allocate_view();

    // Reshape should work with both dynamic and static dims
    auto v1 = f1.get_view<Real[3][24]>();
    auto v2 = f1.get_view<Real**>();

    REQUIRE(v1.size()==v2.size());

    // But if wrong static length is used, we should throw
    REQUIRE_THROWS(f1.get_view<Real[3][16]>());

    // Should not be able to reshape to this data type...
    REQUIRE_THROWS(f1.get_view<P16**>());
    // But this should work
    f1.get_view<P8**>();

    // Using packs (of allowable size) of different pack sizes
    // should lead to views with different extents.
    // Since there's no padding, their extent on last dimension
    // should be the phys dim divided by pack size.
    auto v3 = f1.get_view<P8**>();
    auto v4 = f1.get_view<P4**>();
    REQUIRE (v4.size() == 2*v3.size());
    REQUIRE (v4.extent_int(0) == fid.get_layout().dim(0));
    REQUIRE (v3.extent_int(0) == fid.get_layout().dim(0));
    REQUIRE (v4.extent_int(1) == fid.get_layout().dim(1) / P4::n);
    REQUIRE (v3.extent_int(1) == fid.get_layout().dim(1) / P8::n);

    // The memory spans should be identical
    REQUIRE (v3.impl_map().memory_span()==v4.impl_map().memory_span());

    // Trying to reshape into something that the allocation cannot accommodate should throw
    REQUIRE_THROWS (f1.get_view<P16***>());

    // Can't get non-const data type view from a read-only field
    REQUIRE_THROWS (f1.get_const().get_view<Real**>());
  }

  SECTION ("equivalent") {
    Field f1 (fid), f2(fid);
    f1.allocate_view();
    f2.allocate_view();

    // Check self equivalence
    // get_const returns a copy of self, so equivalent (if already allocated)
    REQUIRE (f1.equivalent(f1.get_const()));
    REQUIRE (f1.equivalent(f1));
    // f1 and f2 have independent views, so they are not equivalent.
    REQUIRE (!f1.equivalent(f2));
  }

  // Check copy constructor
  SECTION ("copy ctor") {
    Field f1 (fid);

    f1.allocate_view();
    f1.deep_copy(3.0);

    Field f2 = f1;
    REQUIRE(f2.get_header_ptr()==f1.get_header_ptr());
    REQUIRE(f2.get_internal_view_data<Real>()==f1.get_internal_view_data<Real>());
    REQUIRE(f2.is_allocated());
    REQUIRE(views_are_equal(f1,f2));
  }

  SECTION ("construct_from_view") {
    // Crate f1 with some padding, to stress test the feature
    Field f1 (fid);
    auto& fap1 = f1.get_header().get_alloc_properties();
    fap1.request_allocation(16);
    f1.allocate_view();
    f1.deep_copy(1.0);

    // Get f1 view, and wrap it in another field
    auto view = f1.get_view<Real**>();
    Field f2 (fid,view);

    // Check the two are the same
    REQUIRE (views_are_equal(f1,f2));

    // Modify one field, and check again
    randomize(f2,engine,pdf);
    REQUIRE (views_are_equal(f1,f2));
  }

  SECTION ("clone") {
    Field f1 (fid);
    auto& fap1 = f1.get_header().get_alloc_properties();

    fap1.request_allocation(16);
    f1.allocate_view();
    f1.deep_copy(3.0);

    Field f2 = f1.clone();
    auto& fap2 = f2.get_header().get_alloc_properties();
    REQUIRE(f2.is_allocated());
    REQUIRE(fap2.get_alloc_size()==fap1.get_alloc_size());
    REQUIRE(views_are_equal(f1,f2));

    // Changing f2 should leave f1 unchanged
    f2.deep_copy<Real>(0.0);
    REQUIRE (field_max<Real>(f2)==0.0);
    REQUIRE (field_min<Real>(f2)==0.0);
    REQUIRE (field_max<Real>(f1)==3.0);
    REQUIRE (field_min<Real>(f1)==3.0);
  }

  SECTION ("alias") {
    Field f1 (fid);
    f1.allocate_view();

    Field f2 = f1.alias("the_alias");

    REQUIRE(f2.is_allocated());
    REQUIRE(&f1.get_header().get_tracking()==&f2.get_header().get_tracking());
    REQUIRE(&f1.get_header().get_alloc_properties()==&f2.get_header().get_alloc_properties());
    REQUIRE(f1.get_header().get_identifier().get_layout()==f2.get_header().get_identifier().get_layout());
    REQUIRE(f1.get_internal_view_data<Real>()==f2.get_internal_view_data<Real>());

    // Identifiers are separate objects though
    REQUIRE(&f1.get_header().get_identifier()!=&f2.get_header().get_identifier());

    // Check extra data is also shared
    f1.get_header().set_extra_data("foo",1);
    REQUIRE (f2.get_header().has_extra_data("foo"));
  }

  SECTION ("deep_copy") {
    // rank-0
    std::vector<FieldTag> t0 = {};
    std::vector<int> d0 = {};
    FieldIdentifier fid0("scalar_0d",{t0,d0},m/s,"some_grid");
    Field f0(fid0);
    f0.allocate_view();
    f0.deep_copy(1.5);
    f0.sync_to_host();
    REQUIRE (reinterpret_cast<Real*>(f0.get_internal_view_data<Real,Host>())[0]==1.5);

    // rank-3
    std::vector<FieldTag> t1 = {COL,CMP,LEV};
    std::vector<int> d1 = {3,2,24};

    FieldIdentifier fid1("vec_3d",{t1,d1},m/s,"some_grid");

    Field f1(fid1);
    f1.allocate_view();
    f1.deep_copy(1.0);
    f1.sync_to_host();
    auto v = reinterpret_cast<Real*>(f1.get_internal_view_data<Real,Host>());
    for (int i=0; i<fid1.get_layout().size(); ++i) {
      REQUIRE (v[i]==1.0);
    }
  }

  SECTION ("host_view") {
    Field f(fid);

    // Views not yet allocated
    REQUIRE_THROWS(f.get_internal_view_data<Real>());
    REQUIRE_THROWS(f.get_internal_view_data<Real,Host>());
    REQUIRE_THROWS(f.sync_to_host());
    REQUIRE_THROWS(f.sync_to_dev());

    f.allocate_view();
    randomize(f,engine,pdf);

    // Get reshaped view on device, and manually create Host mirror
    auto v2d = f.get_view<Real**>();
    auto v2d_hm = Kokkos::create_mirror_view(v2d);
    Kokkos::deep_copy(v2d_hm,v2d);

    // Get reshaped view straight on Host
    auto v2dh = f.get_view<Real**,Host>();

    // The two should match
    for (int i=0; i<dims[0]; ++i) {
      for (int j=0; j<dims[1]; ++j) {
        REQUIRE (v2dh(i,j) == v2d_hm(i,j) );
      }
    }
  }

  SECTION ("rank0_field") {
    // Create 0d field
    FieldIdentifier fid0("f_0d", FieldLayout({},{}), Units::nondimensional(), "dummy_grid");
    Field f0(fid0);
    f0.allocate_view();

    // Create 1d field
    FieldIdentifier fid1("f_1d", FieldLayout({COL}, {5}), Units::nondimensional(), "dummy_grid");
    Field f1(fid1);
    f1.allocate_view();

    // Randomize 1d field
    randomize(f1,engine,pdf);

    auto v0 = f0.get_view<Real, Host>();
    auto v1 = f1.get_view<Real*, Host>();

    // Deep copy subfield of 1d field -> 0d field and check result
    for (size_t i=0; i<v1.extent(0); ++i) {
      f0.deep_copy<Host>(f1.subfield(0, i));
      REQUIRE(v0() == v1(i));
    }

    // Randomize 0d field
    randomize(f0,engine,pdf);

    // Deep copy 0d field -> subfield of 1d field and check result
    for (size_t i=0; i<v1.extent(0); ++i) {
      f1.subfield(0, i).deep_copy<Host>(f0);
      REQUIRE(v1(i) == v0());
    }
  }
}

TEST_CASE("field_group") {
  using namespace scream;
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  using FID = FieldIdentifier;
  using FL  = FieldLayout;

  constexpr int ncols = 10;
  constexpr int ndims = 4;
  constexpr int nlevs = 8;

  FID fid ("V",FL({COL,CMP,LEV},{ncols,ndims,nlevs}),Units::nondimensional(),"the_grid");
  Field f (fid);
  f.allocate_view();

  FieldGroupInfo info("G");
  info.m_bundled = true;
  std::vector<Field> f_i;
  for (int i=0; i<ndims; ++i) {
    f_i.push_back(f.get_component(i));
    info.m_fields_names.push_back(f_i[i].name());
    info.m_subview_dim = 1;
    info.m_subview_idx[f_i[i].name()] = i;
  }

  // Create group and set subfields
  FieldGroup g(info);
  g.m_bundle = std::make_shared<Field>(f);
  for (int i=0; i<ndims; ++i) {
    g.m_fields["G_"+std::to_string(i)] = std::make_shared<Field>(f_i[i]);
  }

  // Check const cloning
  auto cg= g.get_const();
  REQUIRE (cg.m_bundle->is_read_only());
  REQUIRE (cg.m_fields.size()==g.m_fields.size());
  REQUIRE (*cg.m_info==*g.m_info);
  REQUIRE (cg.m_bundle->get_internal_view_data<const Real>()==
            g.m_bundle->get_internal_view_data<const Real>());
  for (int i=0; i<ndims; ++i) {
    const auto&  f =  *g.m_fields.at("G_"+std::to_string(i));
    const auto& cf = *cg.m_fields.at("G_"+std::to_string(i));
    REQUIRE ( f.get_internal_view_data<const Real>()==
             cf.get_internal_view_data<const Real>());
  }
}

TEST_CASE("field_mgr", "") {
  using namespace scream;
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  using FID = FieldIdentifier;
  using FR  = FieldRequest;
  using SL  = std::list<std::string>;
  using Pack1 = ekat::Pack<Real,8>;
  using Pack2 = ekat::Pack<Real,16>;

  const int ncols1 = 4; const int ncols2 = 7;
  const int nlevs1 = 7; const int nlevs2 = 10;
  const int subview_dim = 1;
  const int subview_slice = 0;

  std::vector<FieldTag> tags1 = {COL,LEV};
  std::vector<FieldTag> tags2 = {COL,CMP,LEV};

  std::vector<FieldTag> bad_tags = {EL,GP,GP};

  std::vector<int> dims1 = {ncols1,nlevs1}; std::vector<int> dims2 = {ncols2,nlevs2};
  std::vector<int> dims3 = {ncols1,10,nlevs1}; std::vector<int> dims4 = {ncols2,10,nlevs2};

  std::vector<int> bad_dims = {2,4,4};

  FID fid1_1("field1_1", {tags1, dims1},  m/s, "grid1");
  FID fid1_2("field1_2", {tags1, dims2},  m/s, "grid2");
  FID fid2_1("field2_1", {tags2, dims3},  m/s, "grid1");
  FID fid2_2("field2_2", {tags2, dims4},  m/s, "grid2");

  const auto km = 1000*m;
  FID bad1("field1_1", {tags1, dims1},  m/s, "grid3"); // Bad grid
  FID bad2("field1_1", {tags1, dims1}, km/s, "grid1"); // Bad units
  FID bad3("field1_1", {tags1, dims2},  m/s, "grid1"); // Bad layout

  ekat::Comm comm(MPI_COMM_WORLD);
  auto g1 = create_point_grid("grid1",ncols1*comm.size(),nlevs1,comm);
  auto g2 = create_point_grid("grid2",ncols2*comm.size(),nlevs2,comm);
  auto gm = std::make_shared<LibraryGridsManager>(g1, g2);
  FieldManager field_mgr(gm);

  // Should not be able to register fields yet
  REQUIRE_THROWS(field_mgr.register_field(FR(fid1_1)));

  field_mgr.registration_begins();

  // === Valid registration calls === //
  field_mgr.register_field(FR(fid1_1,Pack1::n));
  field_mgr.register_field(FR{fid1_2,Pack2::n});
  field_mgr.register_field(FR{fid2_1});
  field_mgr.register_field(FR{fid2_1,"group_1"});
  field_mgr.register_field(FR{fid1_2,SL{"group_1", "group_2"}});
  field_mgr.register_field(FR{fid2_2});

  // === Invalid registration calls === //
  REQUIRE_THROWS(field_mgr.register_field(FR{bad1}));
  REQUIRE_THROWS(field_mgr.register_field(FR{bad2}));
  REQUIRE_THROWS(field_mgr.register_field(FR{bad2}));

  // Cannot add external fields while registration is happening
  REQUIRE_THROWS(field_mgr.add_field(Field(fid1_1)));

  field_mgr.registration_ends();

  // Should not be able to register fields anymore
  REQUIRE_THROWS(field_mgr.register_field(FR{fid1_1}));

  FID new_fid("new_field", {tags1, dims1},  m/s, "grid1");
  REQUIRE_THROWS (field_mgr.add_field(Field(new_fid))); // Not allocated

  REQUIRE (field_mgr.size("grid1")==2);
  REQUIRE (field_mgr.size("grid2")==2);

  // Get all fields
  auto f1_1 = field_mgr.get_field(fid1_1);
  auto f1_2 = field_mgr.get_field(fid1_2);
  auto f2_1 = field_mgr.get_field(fid2_1);
  auto f2_2 = field_mgr.get_field(fid2_2);

  // Verify both get_field methods match
  REQUIRE (f1_1 == field_mgr.get_field(fid1_1.name(), fid1_1.get_grid_name()));
  REQUIRE (f1_2 == field_mgr.get_field(fid1_2.name(), fid1_2.get_grid_name()));
  REQUIRE (f2_1 == field_mgr.get_field(fid2_1.name(), fid2_1.get_grid_name()));
  REQUIRE (f2_2 == field_mgr.get_field(fid2_2.name(), fid2_2.get_grid_name()));

  // Try to get invalid fields
  REQUIRE_THROWS(field_mgr.get_field("bad", "grid1"));      // Not in the field_mgr
  REQUIRE_THROWS(field_mgr.get_field(bad1));                // Not in field_mgr
  REQUIRE_THROWS(field_mgr.get_field("field1_2", "grid1")); // Wrong grid

  // Check that the groups names are in the header. While at it, make sure that case insensitive works fine.
  auto has_group = [](const ekat::WeakPtrSet<const FieldGroupInfo>& groups,
                      const std::string& name)->bool {
    for (auto it : groups) {
      if (it.lock()->m_group_name==name) {
        return true;
      }
    }
    return false;
  };
  REQUIRE (has_group(f2_1.get_header().get_tracking().get_groups_info(),"gRouP_1"));
  REQUIRE (has_group(f1_2.get_header().get_tracking().get_groups_info(),"Group_1"));
  REQUIRE (has_group(f1_2.get_header().get_tracking().get_groups_info(),"Group_2"));

  // Check that the groups in the field_mgr contain the correct fields
  REQUIRE (field_mgr.get_groups_info("grid1").count("GROUP_1")==1);
  REQUIRE (field_mgr.get_groups_info("grid2").count("GRoup_1")==1);
  REQUIRE (field_mgr.get_groups_info("grid2").count("group_2")==1);
  REQUIRE (field_mgr.get_groups_info("grid1").count("group_2")==0);
  REQUIRE (field_mgr.get_groups_info("grid2").at("group_1")->m_fields_names.size()==1);

  auto g1_1 = field_mgr.get_groups_info("grid1").at("group_1");
  auto g1_2 = field_mgr.get_groups_info("grid2").at("group_1");
  auto g2_2 = field_mgr.get_groups_info("grid2").at("group_2");
  REQUIRE (ekat::contains(g1_1->m_fields_names,"Field2_1"));
  REQUIRE (ekat::contains(g1_2->m_fields_names,"Field1_2"));
  REQUIRE (ekat::contains(g2_2->m_fields_names,"Field1_2"));

  // Check alloc props for f1 and f2 (which requested pack size > 1)
  auto f1_1_padding = f1_1.get_header().get_alloc_properties().get_padding();
  auto f1_2_padding = f1_2.get_header().get_alloc_properties().get_padding();

  REQUIRE (f1_1_padding==ekat::PackInfo<Pack1::n>::padding(nlevs1));
  REQUIRE (f1_2_padding==ekat::PackInfo<Pack2::n>::padding(nlevs2));

  // Try to subview a field and set the subfield back in the FM
  field_mgr.add_field(f2_1.subfield("field2_1_sf",subview_dim,subview_slice,true));
  REQUIRE (field_mgr.size("grid1")==3);

  auto f2_1_sf = field_mgr.get_field("field2_1_sf", "grid1");
  REQUIRE_THROWS (field_mgr.add_field(f2_1_sf)); // Cannot have duplicates
}

TEST_CASE("tracers_bundle", "") {
  using namespace scream;
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  using FR  = FieldRequest;

  const int ncols1 = 4;
  const int ncols2 = 3;
  const int nlevs = 7;

  std::vector<FieldTag> tags = {COL,LEV};
  std::vector<int> dims1 = {ncols1,nlevs};
  std::vector<int> dims2 = {ncols2,nlevs};

  const auto nondim = Units::nondimensional();

  const std::string gn1 = "grid1";
  const std::string gn2 = "grid2";

  FieldIdentifier qv_id("qv", {tags, dims1}, nondim, gn1);
  FieldIdentifier qc_id("qc", {tags, dims1}, nondim, gn1);
  FieldIdentifier qr_id("qr", {tags, dims2}, nondim, gn2);

  FieldIdentifier a1_id("a1", {tags, dims1}, nondim, gn1);
  FieldIdentifier a2_id("a2", {tags, dims2}, nondim, gn2);
  //FieldIdentifier _a2_id("a2", {tags, dims1}, nondim, gn1);
  FieldIdentifier a3_id("a3", {tags, dims1}, nondim, gn1);

  ekat::Comm comm(MPI_COMM_WORLD);
  auto g1 = create_point_grid(gn1,ncols1*comm.size(),nlevs,comm);
  auto g2 = create_point_grid(gn2,ncols2*comm.size(),nlevs,comm);
  auto gm = std::make_shared<LibraryGridsManager>(g1, g2);
  FieldManager field_mgr(gm);

  field_mgr.registration_begins();

  using los = std::list<std::string>;
  field_mgr.register_field(FR{qv_id,los{"tracers", "turbulence_advected_tracers"}});
  field_mgr.register_field(FR{qc_id,los{"tracers", "turbulence_advected_tracers"}});
  field_mgr.register_field(FR{qr_id,los{"tracers", "turbulence_advected_tracers"}});
  field_mgr.register_field(FR{a1_id,"tracers"});
  field_mgr.register_field(FR{a2_id,"tracers"});
  //field_mgr.register_field(FR{_a2_id,"tracers"});
  field_mgr.register_field(FR{a3_id,"tracers"});
  field_mgr.register_group(GroupRequest("tracers",gn1,Bundling::Required));
  field_mgr.register_group(GroupRequest("turbulence_advected_tracers",gn1,Bundling::Required));
  field_mgr.register_group(GroupRequest("tracers",gn2,Bundling::Required));

  field_mgr.registration_ends();

  // The field_mgr should have allocated the group bundled
  auto tracers1 = field_mgr.get_field_group("tracers", gn1);
  auto tracers2 = field_mgr.get_field_group("tracers", gn2);
  auto turb_tracers = field_mgr.get_field_group("turbulence_advected_tracers", gn1);
  REQUIRE (tracers1.m_info->m_bundled);
  REQUIRE (tracers2.m_info->m_bundled);
  REQUIRE (turb_tracers.m_info->m_bundled);


  printf("Tracers1:");
  for (auto fn : tracers1.m_info->m_fields_names) {
    printf(" %s", fn.c_str());
  }
  printf("\nTurbTracers1:");
  for (auto fn : turb_tracers.m_info->m_fields_names) {
    printf(" %s", fn.c_str());
  }
  printf("\nTracers2:");
  for (auto fn : tracers2.m_info->m_fields_names) {
    printf(" %s", fn.c_str());
  }
  printf("\n");



  // REQUIRE ((qv_p!=nullptr && qv_p.get()==&Q.get_header()));
  // REQUIRE ((qc_p!=nullptr && qc_p.get()==&Q.get_header()));
  // REQUIRE ((qr_p!=nullptr && qr_p.get()==&Q.get_header()));


  // REQUIRE ((a1_p!=nullptr && a1_p.get()==&Q_turb.get_header()));
  // REQUIRE ((a2_p!=nullptr && a2_p.get()==&Q_turb.get_header()));
  // REQUIRE ((a3_p!=nullptr && a3_p.get()==&Q_turb.get_header()));

  // // The indices used for each q to subview Q
  // int idx_v, idx_c, idx_r, idx_a1, idx_a2, idx_a3;

  // // The idx must be stored
  // idx_v = group.m_info->m_subview_idx.at("qv");
  // idx_c = group.m_info->m_subview_idx.at("qc");
  // idx_r = group.m_info->m_subview_idx.at("qr");
  // idx_a1 = group_turb.m_info->m_subview_idx.at("a1");
  // idx_a2 = group_turb.m_info->m_subview_idx.at("a2");
  // idx_a3 = group_turb.m_info->m_subview_idx.at("a3");

  // // For each idx must be in [0,2] and must be different
  // REQUIRE ((idx_v>=0 && idx_v<3 &&
  //           idx_c>=0 && idx_c<3 &&
  //           idx_r>=0 && idx_r<3));
  // REQUIRE ((idx_a1>=0 && idx_a1<3 &&
  //           idx_a2>=0 && idx_a2<3 &&
  //           idx_a3>=0 && idx_a3<3));
  // REQUIRE ((idx_v!=idx_c && idx_v!=idx_r && idx_c!=idx_r));

  // // Now fill Q with random values
  // auto engine = setup_random_test(&comm);
  // using RPDF = std::uniform_real_distribution<Real>;
  // RPDF pdf(0.0,1.0);

  // randomize(Q,engine,pdf);

  // // Check that the same values are in all q's
  // Q.sync_to_host();
  // auto Qh = Q.get_view<Real***,Host>();
  // auto qvh = qv.get_view<Real**,Host>();
  // auto qch = qc.get_view<Real**,Host>();
  // auto qrh = qr.get_view<Real**,Host>();

  // for (int icol=0; icol<ncols1; ++icol) {
  //   for (int ilev=0; ilev<nlevs; ++ilev) {
  //     REQUIRE (Qh(icol,idx_v,ilev)==qvh(icol,ilev));
  //     REQUIRE (Qh(icol,idx_c,ilev)==qch(icol,ilev));
  //     REQUIRE (Qh(icol,idx_r,ilev)==qrh(icol,ilev));
  //   }
  // }

  // // Check that the field ptrs stored in the group are the same as the q
  // auto qv_ptr = group.m_fields.at("qv");
  // auto qc_ptr = group.m_fields.at("qc");
  // auto qr_ptr = group.m_fields.at("qr");

  // REQUIRE (qv_ptr->equivalent(qv));
  // REQUIRE (qc_ptr->equivalent(qc));
  // REQUIRE (qr_ptr->equivalent(qr));
}

TEST_CASE ("update") {
  using namespace scream;
  using namespace ekat::units;

  using namespace ShortFieldTagsNames;
  using RPDF = std::uniform_real_distribution<Real>;
  using IPDF = std::uniform_int_distribution<int>;

  // Setup random number generation
  ekat::Comm comm(MPI_COMM_WORLD);
  auto engine = setup_random_test ();
  RPDF rpdf(0,1);
  IPDF ipdf(0,100);

  const int ncol = 2;
  const int ncmp = 3;
  const int nlev = 4;

  // Create field (if available, use packs, to ensure we don't print garbage)
  std::vector<FieldTag> tags = {COL, CMP, LEV};
  std::vector<int>      dims = {ncol,ncmp,nlev};

  FieldIdentifier fid_r ("fr", {tags,dims}, kg, "some_grid", DataType::RealType);
  FieldIdentifier fid_i ("fi", {tags,dims}, kg, "some_grid", DataType::IntType);
  Field f_real (fid_r);
  Field f_int  (fid_i);
  f_real.allocate_view();
  f_int.allocate_view();
  randomize (f_real,engine,rpdf);
  randomize (f_int, engine,ipdf);

  SECTION ("data_type_checks") {
    Field f2 = f_int.clone();

    // Coeffs have wrong data type (precision loss casting to field's data type)
    REQUIRE_THROWS (f2.update (f_int,1.0,1.0));

    // RHS has wrong data type
    REQUIRE_THROWS (f2.update(f_real,1,0));
  }

  SECTION ("deep_copy") {
    Field f2 (fid_r);
    f2.allocate_view();

    // Replace f2's content with f_real's content
    f2.update(f_real,1,0);
    REQUIRE (views_are_equal(f2,f_real));
  }

  SECTION ("update") {
    Field f2 = f_real.clone();
    Field f3 = f_real.clone();

    // x+x == 2*x
    f2.update(f_real,1,1);
    f3.scale(2);
    REQUIRE (views_are_equal(f2,f3));

    // Adding 2*f_real to N*f3 should give 2*f_real (f3==0)
    f3.deep_copy(0.0);
    f3.update(f_real,2,10);
    REQUIRE (views_are_equal(f3,f2));

    // Same, but we discard current content of f3
    f3.update(f_real,2,0);
    REQUIRE (views_are_equal(f3,f2));
  }

  SECTION ("scale") {
    Field f1 = f_real.clone();
    Field f2 = f_real.clone();

    // x=2, x*y = 2*y
    f1.deep_copy(2.0);
    f1.scale(f2);
    f2.scale(2.0);
    REQUIRE (views_are_equal(f1, f2));
  }
}


TEST_CASE ("sync_subfields") {
  // This test is for previously incorrect behavior, where syncing a subfield
  // to host/device would deep copy the entire data view (including all entries of
  // the parent view). Here, if memory space is not shared between host and device,
  // syncing a subfield to host/device will not sync the data of the other subfields.

  using namespace scream;
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  using FID = FieldIdentifier;
  using FL  = FieldLayout;

  constexpr int ncols = 10;
  constexpr int ndims = 4;
  constexpr int nlevs = 8;

  // Create field with (col, cmp, lev)
  FID fid ("V",FL({COL,CMP,LEV},{ncols,ndims,nlevs}),Units::nondimensional(),"the_grid",DataType::IntType);
  Field f (fid);
  f.allocate_view();

  // Store whether mem space for host and device are the same for testing subfield values
  const bool shared_mem_space = f.host_and_device_share_memory_space();

  // Deep copy all values to ndims on device and host
  f.deep_copy(ndims);
  f.sync_to_host();

  // Set subfield values to their index on device
  for (int c=0; c<ndims; ++c) {
    f.get_component(c).deep_copy(c);
  }

  // Sync only component 0 to host
  f.get_component(0).sync_to_host();

  // For components 1,...,ndims-1, if device and host do not share a
  // memory space, host values should be equal to ndims, else host
  // values should be equal to component index
  for (int c=1; c<ndims; ++c) {
    auto host_subview = f.get_component(c).get_view<int**, Host>();
    for (int idx=0; idx<ncols*nlevs; ++idx) {
      const int icol = idx/nlevs; const int ilev = idx%nlevs;
      if (shared_mem_space) REQUIRE(host_subview(icol, ilev) == c);
      else                  REQUIRE(host_subview(icol, ilev) == ndims);
    }
  }

  // Deep copy all values to ndims on device and host
  f.deep_copy(ndims);
  f.sync_to_host();

  // Set subfield values to their index on host
  for (int c=0; c<ndims; ++c) {
    f.get_component(c).deep_copy<int, Host>(c);
  }

  // Sync only component 0 to device
  f.get_component(0).sync_to_dev();

  // For components 1,...,ndims-1, if device and host do not share a
  // memory space, device values should be equal to ndims, else device
  // values should be equal to component index
  for (int c=1; c<ndims; ++c) {
    auto device_subview = f.get_component(c).get_view<int**, Device>();
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ncols,nlevs}),
                         KOKKOS_LAMBDA (const int icol, const int ilev) {
      if (shared_mem_space) EKAT_KERNEL_ASSERT(device_subview(icol, ilev) == c);
      else                  EKAT_KERNEL_ASSERT(device_subview(icol, ilev) == ndims);
    });
  }
}

} // anonymous namespace
